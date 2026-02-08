"""
OptionsTitan Input Panel
User input widgets with validation
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, 
    QLineEdit, QDoubleSpinBox, QComboBox, QPushButton,
    QGroupBox, QLabel
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QValidator, QDoubleValidator, QRegularExpressionValidator
import re


class StockSymbolValidator(QValidator):
    """Validator for stock symbols (uppercase, 1-5 characters)"""
    
    def validate(self, text, pos):
        text = text.upper()
        
        if not text:
            return QValidator.Intermediate, text, pos
        
        # Check if valid stock symbol pattern
        if re.match(r'^[A-Z]{1,5}$', text):
            return QValidator.Acceptable, text, pos
        elif re.match(r'^[A-Z]{0,5}$', text):
            return QValidator.Intermediate, text, pos
        else:
            return QValidator.Invalid, text, pos


class InputPanel(QWidget):
    """
    Input panel widget for strategy analysis parameters.
    Provides input fields with validation and helpful tooltips.
    """
    
    # Signal emitted when user wants to analyze
    analyze_requested = Signal(str, float, float, float, float)  # symbol, liquidity, risk, profit, loss
    clear_requested = Signal()
    
    def __init__(self):
        super().__init__()
        self.setup_ui()
        self.setup_connections()
    
    def setup_ui(self):
        """Create the compact dashboard-style UI layout"""
        layout = QVBoxLayout(self)
        layout.setSpacing(20)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Create input group with dashboard styling
        input_group = QGroupBox("Strategy Parameters")
        input_group.setProperty("dashboard_card", True)
        input_layout = QVBoxLayout(input_group)
        input_layout.setSpacing(18)
        input_layout.setContentsMargins(20, 25, 20, 20)
        
        # Stock Symbol
        symbol_label = QLabel("Stock Symbol")
        symbol_label.setProperty("input_label", True)
        input_layout.addWidget(symbol_label)
        
        self.symbol_combo = QComboBox()
        self.symbol_combo.setEditable(True)
        self.symbol_combo.setMaxVisibleItems(10)
        self.symbol_combo.addItems([
            "SPY", "QQQ", "IWM", "DIA", "AAPL", "MSFT", "GOOGL", 
            "AMZN", "TSLA", "NVDA", "META", "AMD"
        ])
        self.symbol_combo.setCurrentText("SPY")
        self.symbol_combo.setToolTip(
            "Enter or select a stock symbol (e.g., SPY, AAPL, TSLA)\n"
            "Popular symbols are pre-loaded in the dropdown."
        )
        # Add validator to the line edit inside combo box
        self.symbol_combo.lineEdit().setValidator(StockSymbolValidator())
        input_layout.addWidget(self.symbol_combo)
        
        # Portfolio Liquidity
        liquidity_label = QLabel("Portfolio Liquidity")
        liquidity_label.setProperty("input_label", True)
        input_layout.addWidget(liquidity_label)
        
        self.liquidity_spin = QDoubleSpinBox()
        self.liquidity_spin.setRange(1000, 10000000)
        self.liquidity_spin.setValue(10000)
        self.liquidity_spin.setPrefix("$ ")
        self.liquidity_spin.setDecimals(2)
        self.liquidity_spin.setSingleStep(1000)
        self.liquidity_spin.setToolTip(
            "Your available trading capital.\n"
            "This determines position sizing and which strategies are feasible.\n"
            "Example: $10,000"
        )
        input_layout.addWidget(self.liquidity_spin)
        
        # Max Risk %
        risk_label = QLabel("Max Risk (%)")
        risk_label.setProperty("input_label", True)
        input_layout.addWidget(risk_label)
        
        self.risk_spin = QDoubleSpinBox()
        self.risk_spin.setRange(0.1, 100)
        self.risk_spin.setValue(5.0)
        self.risk_spin.setSuffix(" %")
        self.risk_spin.setDecimals(1)
        self.risk_spin.setSingleStep(0.5)
        self.risk_spin.setToolTip(
            "Maximum percentage of your portfolio to risk per trade.\n"
            "Conservative: 1-3%, Moderate: 3-7%, Aggressive: 7%+\n"
            "Recommended: 2-5% for options trading"
        )
        input_layout.addWidget(self.risk_spin)
        
        # Target Profit %
        profit_label = QLabel("Target Profit (%)")
        profit_label.setProperty("input_label", True)
        input_layout.addWidget(profit_label)
        
        self.profit_spin = QDoubleSpinBox()
        self.profit_spin.setRange(1, 1000)
        self.profit_spin.setValue(20.0)
        self.profit_spin.setSuffix(" %")
        self.profit_spin.setDecimals(1)
        self.profit_spin.setSingleStep(5)
        self.profit_spin.setToolTip(
            "Your target profit percentage for the trade.\n"
            "This helps the analyzer recommend appropriate strategies.\n"
            "Typical range: 15-50% for options"
        )
        input_layout.addWidget(self.profit_spin)
        
        # Max Loss %
        loss_label = QLabel("Max Loss (%)")
        loss_label.setProperty("input_label", True)
        input_layout.addWidget(loss_label)
        
        self.loss_spin = QDoubleSpinBox()
        self.loss_spin.setRange(1, 100)
        self.loss_spin.setValue(15.0)
        self.loss_spin.setSuffix(" %")
        self.loss_spin.setDecimals(1)
        self.loss_spin.setSingleStep(5)
        self.loss_spin.setToolTip(
            "Maximum loss you're willing to accept.\n"
            "This sets your stop-loss level.\n"
            "Should typically be less than your max risk per trade."
        )
        input_layout.addWidget(self.loss_spin)
        
        layout.addWidget(input_group)
        
        # Action Buttons
        self.analyze_button = QPushButton("🔍 Analyze Strategies")
        self.analyze_button.setProperty("primary_action", True)
        self.analyze_button.setMinimumHeight(55)
        self.analyze_button.setCursor(Qt.PointingHandCursor)
        self.analyze_button.setToolTip(
            "Run the analysis to get personalized strategy recommendations"
        )
        layout.addWidget(self.analyze_button)
        
        # Secondary buttons row
        secondary_layout = QHBoxLayout()
        secondary_layout.setSpacing(10)
        
        self.clear_button = QPushButton("🗑️ Clear")
        self.clear_button.setProperty("secondary", True)
        self.clear_button.setMinimumHeight(45)
        self.clear_button.setCursor(Qt.PointingHandCursor)
        self.clear_button.setToolTip("Clear the analysis results")
        
        self.defaults_button = QPushButton("↺ Reset")
        self.defaults_button.setProperty("secondary", True)
        self.defaults_button.setMinimumHeight(45)
        self.defaults_button.setCursor(Qt.PointingHandCursor)
        self.defaults_button.setToolTip("Reset all inputs to default values")
        
        secondary_layout.addWidget(self.clear_button)
        secondary_layout.addWidget(self.defaults_button)
        
        layout.addLayout(secondary_layout)
    
    def setup_connections(self):
        """Connect signals and slots"""
        self.analyze_button.clicked.connect(self.on_analyze_clicked)
        self.clear_button.clicked.connect(self.clear_requested.emit)
        self.defaults_button.clicked.connect(self.reset_to_defaults)
    
    def on_analyze_clicked(self):
        """Handle analyze button click"""
        # Validate inputs
        symbol = self.symbol_combo.currentText().strip().upper()
        
        if not symbol:
            return
        
        # Get values
        liquidity = self.liquidity_spin.value()
        max_risk = self.risk_spin.value()
        target_profit = self.profit_spin.value()
        max_loss = self.loss_spin.value()
        
        # Emit signal with values
        self.analyze_requested.emit(symbol, liquidity, max_risk, target_profit, max_loss)
    
    def reset_to_defaults(self):
        """Reset all inputs to default values"""
        self.symbol_combo.setCurrentText("SPY")
        self.liquidity_spin.setValue(10000)
        self.risk_spin.setValue(5.0)
        self.profit_spin.setValue(20.0)
        self.loss_spin.setValue(15.0)
    
    def set_enabled(self, enabled):
        """Enable or disable all input widgets"""
        self.symbol_combo.setEnabled(enabled)
        self.liquidity_spin.setEnabled(enabled)
        self.risk_spin.setEnabled(enabled)
        self.profit_spin.setEnabled(enabled)
        self.loss_spin.setEnabled(enabled)
        self.analyze_button.setEnabled(enabled)
        self.clear_button.setEnabled(enabled)
        self.defaults_button.setEnabled(enabled)
    
    def get_values(self):
        """Get all input values as a dictionary"""
        return {
            'symbol': self.symbol_combo.currentText().strip().upper(),
            'liquidity': self.liquidity_spin.value(),
            'max_risk': self.risk_spin.value(),
            'target_profit': self.profit_spin.value(),
            'max_loss': self.loss_spin.value()
        }
