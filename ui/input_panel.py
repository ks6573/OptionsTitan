"""
OptionsTitan Input Panel
User input widgets with validation
"""

import json
from pathlib import Path

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLineEdit, QDoubleSpinBox, QComboBox, QPushButton,
    QGroupBox, QLabel, QCompleter
)
from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QValidator
import re


def _responsive_values():
    """Get responsive sizing values. Lazy import to avoid circular deps."""
    try:
        from .ui_utils import (
            get_screen_size,
            get_screen_percentage_height,
            get_responsive_spacing,
        )
        width, height = get_screen_size()
        spacing = get_responsive_spacing()
        analyze_h = max(46, min(64, int(height * 0.058)))
        secondary_h = max(38, min(52, int(height * 0.048)))
        chip_w = max(52, min(72, int(width * 0.032)))
        margins = max(16, min(28, int(width * 0.012)))
        return spacing, analyze_h, secondary_h, chip_w, margins
    except Exception:
        return 18, 54, 46, 54, 24

# Favorites for Quick Select
FAVORITES = ["SPY", "QQQ", "IWM", "DIA", "AAPL", "MSFT", "TSLA", "NVDA", "META", "AMD"]
RECENT_SEARCHES_PATH = Path("data/cache/recent_searches.json")
MAX_RECENT = 20


def _load_recent_searches() -> list:
    """Load recent search symbols from cache."""
    if not RECENT_SEARCHES_PATH.exists():
        return []
    try:
        with open(RECENT_SEARCHES_PATH, encoding="utf-8") as f:
            data = json.load(f)
        return list(data.get("symbols", []))[:MAX_RECENT]
    except (json.JSONDecodeError, OSError):
        return []


def _save_recent_search(symbol: str):
    """Append symbol to recent searches and persist."""
    recents = _load_recent_searches()
    sym = (symbol or "").strip().upper()
    if not sym:
        return
    recents = [s for s in recents if s.upper() != sym]
    recents.insert(0, sym)
    recents = recents[:MAX_RECENT]
    RECENT_SEARCHES_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(RECENT_SEARCHES_PATH, "w", encoding="utf-8") as f:
            json.dump({"symbols": recents}, f, indent=0)
    except OSError:
        pass


class StockSymbolValidator(QValidator):
    """Validator for stock symbols (uppercase, 1-10 chars, letters, digits, dot for BRK.B)"""

    def validate(self, text, pos):
        text = text.upper()

        if not text:
            return QValidator.Intermediate, text, pos

        if re.match(r"^[A-Z0-9.]{1,10}$", text):
            return QValidator.Acceptable, text, pos
        if re.match(r"^[A-Z0-9.]{0,10}$", text):
            return QValidator.Intermediate, text, pos
        return QValidator.Invalid, text, pos


class InputPanel(QWidget):
    """
    Input panel widget for strategy analysis parameters.
    Provides input fields with validation and helpful tooltips.
    """

    analyze_requested = Signal(str, float, float, float, float)
    clear_requested = Signal()
    validation_failed = Signal(str, str)  # symbol, error_message

    def __init__(self):
        super().__init__()
        self._coverage_worker = None
        self._validation_worker = None
        self._coverage_debounce_timer = QTimer(self)
        self._coverage_debounce_timer.setSingleShot(True)
        self._coverage_debounce_timer.timeout.connect(self._run_coverage_check)
        self._completer_symbols = set(FAVORITES) | set(_load_recent_searches())
        self._catalog_loaded = False
        self.setup_ui()
        self.setup_connections()
        self._start_catalog_load()

    def setup_ui(self):
        """Create the compact dashboard-style UI layout"""
        spacing, analyze_h, secondary_h, chip_w, margins = _responsive_values()

        layout = QVBoxLayout(self)
        layout.setSpacing(spacing)
        layout.setContentsMargins(0, 0, 0, 0)

        input_group = QGroupBox("Strategy Parameters")
        input_group.setProperty("dashboard_card", True)
        input_layout = QVBoxLayout(input_group)
        input_layout.setSpacing(max(12, spacing - 2))
        input_layout.setContentsMargins(margins, margins + 4, margins, margins)

        # Stock Symbol row
        symbol_label = QLabel("Stock Symbol")
        symbol_label.setProperty("input_label", True)
        input_layout.addWidget(symbol_label)

        symbol_row = QHBoxLayout()
        symbol_row.setSpacing(8)
        self.symbol_combo = QComboBox()
        self.symbol_combo.setEditable(True)
        self.symbol_combo.setMaxVisibleItems(10)
        self.symbol_combo.addItems(FAVORITES)
        self.symbol_combo.setCurrentText("SPY")
        self.symbol_combo.setToolTip(
            "Enter any stock symbol (e.g., SPY, AAPL, BRK.B)\n"
            "Autocomplete from dataset tickers and recent searches."
        )
        le = self.symbol_combo.lineEdit()
        le.setValidator(StockSymbolValidator())
        le.textChanged.connect(self._on_symbol_text_changed)
        symbol_row.addWidget(self.symbol_combo, 1)

        # Coverage badge
        self.coverage_badge = QLabel("")
        try:
            from .ui_utils import get_responsive_font_size
            fs = get_responsive_font_size('small')
            self.coverage_badge.setStyleSheet(
                f"font-size: {fs}px; padding: 6px 10px; border-radius: 4px;"
            )
        except Exception:
            self.coverage_badge.setStyleSheet(
                "font-size: 13px; padding: 6px 10px; border-radius: 4px;"
            )
        self.coverage_badge.setToolTip(
            "Options history available: Dataset has historical options data.\n"
            "Underlying-only mode: Live/underlying data only."
        )
        symbol_row.addWidget(self.coverage_badge)
        input_layout.addLayout(symbol_row)

        # Quick Select chips (2 rows of 4 to fit sidebar, with spacing)
        quick_label = QLabel("Quick Select")
        quick_label.setProperty("input_label", True)
        try:
            from .ui_utils import get_responsive_font_size
            fs = get_responsive_font_size('body')
            quick_label.setStyleSheet(f"font-size: {fs}px; margin-top: 6px; margin-bottom: 4px;")
        except Exception:
            quick_label.setStyleSheet("font-size: 14px; margin-top: 6px; margin-bottom: 4px;")
        input_layout.addWidget(quick_label)

        chip_spacing = max(10, spacing // 2)
        chips_layout = QGridLayout()
        chips_layout.setHorizontalSpacing(chip_spacing)
        chips_layout.setVerticalSpacing(chip_spacing)
        for i, fav in enumerate(FAVORITES[:8]):
            btn = QPushButton(fav)
            btn.setMinimumWidth(chip_w)
            btn.setMinimumHeight(max(34, secondary_h - 6))
            btn.setProperty("chip", True)
            btn.setCursor(Qt.PointingHandCursor)
            btn.clicked.connect(lambda checked, s=fav: self._set_symbol(s))
            row, col = i // 4, i % 4
            chips_layout.addWidget(btn, row, col)
        input_layout.addLayout(chips_layout)

        # Portfolio Liquidity
        input_layout.addWidget(QLabel("Portfolio Liquidity"))
        self.liquidity_spin = QDoubleSpinBox()
        self.liquidity_spin.setRange(1000, 10000000)
        self.liquidity_spin.setValue(10000)
        self.liquidity_spin.setPrefix("$ ")
        self.liquidity_spin.setDecimals(2)
        self.liquidity_spin.setSingleStep(1000)
        self.liquidity_spin.setToolTip(
            "Your available trading capital."
        )
        input_layout.addWidget(self.liquidity_spin)

        # Max Risk %
        input_layout.addWidget(QLabel("Max Risk (%)"))
        self.risk_spin = QDoubleSpinBox()
        self.risk_spin.setRange(0.1, 100)
        self.risk_spin.setValue(5.0)
        self.risk_spin.setSuffix(" %")
        self.risk_spin.setDecimals(1)
        self.risk_spin.setSingleStep(0.5)
        input_layout.addWidget(self.risk_spin)

        # Target Profit %
        input_layout.addWidget(QLabel("Target Profit (%)"))
        self.profit_spin = QDoubleSpinBox()
        self.profit_spin.setRange(1, 1000)
        self.profit_spin.setValue(20.0)
        self.profit_spin.setSuffix(" %")
        self.profit_spin.setDecimals(1)
        self.profit_spin.setSingleStep(5)
        input_layout.addWidget(self.profit_spin)

        # Max Loss %
        input_layout.addWidget(QLabel("Max Loss (%)"))
        self.loss_spin = QDoubleSpinBox()
        self.loss_spin.setRange(1, 100)
        self.loss_spin.setValue(15.0)
        self.loss_spin.setSuffix(" %")
        self.loss_spin.setDecimals(1)
        self.loss_spin.setSingleStep(5)
        input_layout.addWidget(self.loss_spin)

        layout.addWidget(input_group)

        # Action Buttons
        self.analyze_button = QPushButton("🔍 Analyze Strategies")
        self.analyze_button.setProperty("primary_action", True)
        self.analyze_button.setMinimumHeight(analyze_h)
        self.analyze_button.setCursor(Qt.PointingHandCursor)
        layout.addWidget(self.analyze_button)

        secondary_layout = QHBoxLayout()
        secondary_layout.setSpacing(spacing // 2)
        self.clear_button = QPushButton("🗑️ Clear")
        self.clear_button.setProperty("secondary", True)
        self.clear_button.setMinimumHeight(secondary_h)
        self.clear_button.setCursor(Qt.PointingHandCursor)
        self.defaults_button = QPushButton("↺ Reset")
        self.defaults_button.setProperty("secondary", True)
        self.defaults_button.setMinimumHeight(secondary_h)
        self.defaults_button.setCursor(Qt.PointingHandCursor)
        secondary_layout.addWidget(self.clear_button)
        secondary_layout.addWidget(self.defaults_button)
        layout.addLayout(secondary_layout)

        self._update_completer()
        self._update_coverage_badge("")

    def setup_connections(self):
        self.analyze_button.clicked.connect(self.on_analyze_clicked)
        self.clear_button.clicked.connect(self.clear_requested.emit)
        self.defaults_button.clicked.connect(self.reset_to_defaults)

    def _start_catalog_load(self):
        """Load dataset catalog tickers in background for autocomplete."""
        from .workers import CatalogLoaderWorker
        worker = CatalogLoaderWorker()
        worker.catalog_loaded.connect(self._on_catalog_loaded)
        worker.finished.connect(worker.deleteLater)
        worker.start()
        worker.setParent(self)

    def _on_catalog_loaded(self, tickers: set):
        self._completer_symbols |= {str(t).upper() for t in tickers if t}
        self._catalog_loaded = True
        self._update_completer()

    def _update_completer(self):
        symbols = sorted(self._completer_symbols, key=lambda s: (s not in FAVORITES, s))
        completer = QCompleter(symbols)
        completer.setCaseSensitivity(Qt.CaseInsensitive)
        completer.setFilterMode(Qt.MatchContains)
        self.symbol_combo.setCompleter(completer)

    def _set_symbol(self, symbol: str):
        self.symbol_combo.setCurrentText(symbol)
        self._on_symbol_text_changed(symbol)

    def _on_symbol_text_changed(self, text: str):
        sym = (text or "").strip()
        self._coverage_debounce_timer.stop()
        if not sym:
            self._update_coverage_badge("")
            return
        self._coverage_debounce_timer.start(400)

    def _run_coverage_check(self):
        sym = self.symbol_combo.currentText().strip()
        if not sym:
            self._update_coverage_badge("")
            return
        if self._coverage_worker and self._coverage_worker.isRunning():
            return
        from .workers import CoverageCheckWorker
        self._coverage_worker = CoverageCheckWorker(sym)
        self._coverage_worker.coverage_result.connect(self._on_coverage_result)
        self._coverage_worker.finished.connect(lambda: setattr(self, "_coverage_worker", None))
        self._coverage_worker.start()

    def _on_coverage_result(self, symbol: str, has_coverage: bool):
        current = self.symbol_combo.currentText().strip()
        if current.upper() != symbol.upper():
            return
        try:
            from .ui_utils import get_responsive_font_size
            fs = get_responsive_font_size('small')
        except Exception:
            fs = 13
        base_css = f"font-size: {fs}px; padding: 6px 10px; border-radius: 4px;"
        if has_coverage:
            self.coverage_badge.setText("Options history available")
            self.coverage_badge.setStyleSheet(
                f"{base_css} background: #1a472a; color: #4ade80;"
            )
        else:
            self.coverage_badge.setText("Underlying-only mode")
            self.coverage_badge.setStyleSheet(
                f"{base_css} background: #3d3d3d; color: #94a3b8;"
            )

    def _update_coverage_badge(self, _dummy: str):
        self.coverage_badge.setText("")
        try:
            from .ui_utils import get_responsive_font_size
            fs = get_responsive_font_size('small')
        except Exception:
            fs = 13
        self.coverage_badge.setStyleSheet(f"font-size: {fs}px; padding: 6px 10px;")

    def on_analyze_clicked(self):
        symbol = self.symbol_combo.currentText().strip().upper()
        if not symbol:
            self.validation_failed.emit(symbol, "Please enter a stock symbol.")
            return

        if self._validation_worker and self._validation_worker.isRunning():
            return

        from .workers import TickerValidationWorker
        self._validation_worker = TickerValidationWorker(symbol)
        self._validation_worker.validation_passed.connect(self._on_validation_passed)
        self._validation_worker.validation_failed.connect(self._on_validation_failed)
        self._validation_worker.finished.connect(
            lambda: setattr(self, "_validation_worker", None)
        )
        self.input_panel_set_enabled(False)
        self._validation_worker.start()

    def _on_validation_passed(self, symbol: str):
        liquidity = self.liquidity_spin.value()
        max_risk = self.risk_spin.value()
        target_profit = self.profit_spin.value()
        max_loss = self.loss_spin.value()
        _save_recent_search(symbol)
        self._completer_symbols.add(symbol)
        self._update_completer()
        self.analyze_requested.emit(symbol, liquidity, max_risk, target_profit, max_loss)
        # Main window will disable panel during analysis; no re-enable here

    def _on_validation_failed(self, symbol: str, msg: str):
        self.validation_failed.emit(symbol, msg)
        self.input_panel_set_enabled(True)

    def input_panel_set_enabled(self, enabled: bool):
        """Internal: enable/disable during validation (before analysis starts)."""
        self.symbol_combo.setEnabled(enabled)
        self.liquidity_spin.setEnabled(enabled)
        self.risk_spin.setEnabled(enabled)
        self.profit_spin.setEnabled(enabled)
        self.loss_spin.setEnabled(enabled)
        self.analyze_button.setEnabled(enabled)
        self.clear_button.setEnabled(enabled)
        self.defaults_button.setEnabled(enabled)

    def reset_to_defaults(self):
        self.symbol_combo.setCurrentText("SPY")
        self.liquidity_spin.setValue(10000)
        self.risk_spin.setValue(5.0)
        self.profit_spin.setValue(20.0)
        self.loss_spin.setValue(15.0)
        self._on_symbol_text_changed("SPY")

    def set_enabled(self, enabled):
        self.input_panel_set_enabled(enabled)

    def get_values(self):
        return {
            "symbol": self.symbol_combo.currentText().strip().upper(),
            "liquidity": self.liquidity_spin.value(),
            "max_risk": self.risk_spin.value(),
            "target_profit": self.profit_spin.value(),
            "max_loss": self.loss_spin.value(),
        }

    def has_dataset_coverage(self) -> bool:
        """Check if current symbol has options history in dataset (sync, may block)."""
        sym = self.symbol_combo.currentText().strip()
        if not sym:
            return False
        try:
            from src.data_collection.dataset_catalog import has_dataset_coverage
            return has_dataset_coverage(sym)
        except Exception:
            return False
