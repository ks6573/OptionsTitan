"""
OptionsTitan Strategy Card Widget
Displays individual strategy information in an expandable card format
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QPushButton, QFrame, QTextEdit, QApplication
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont, QClipboard


class StrategyCard(QFrame):
    """
    A card widget displaying a single strategy's details.
    Can be expanded/collapsed to show full information.
    """
    
    copy_requested = Signal(str)  # Emits strategy text to copy
    
    def __init__(self, strategy, rank, parent=None):
        super().__init__(parent)
        self.strategy = strategy
        self.rank = rank
        self.is_expanded = False
        
        self.setup_ui()
    
    def setup_ui(self):
        """Create the card UI"""
        self.setObjectName("strategyCard")
        self.setFrameShape(QFrame.StyledPanel)
        self.setFrameShadow(QFrame.Raised)
        self.setCursor(Qt.PointingHandCursor)
        
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        
        # Header - always visible
        header_layout = self.create_header()
        layout.addLayout(header_layout)
        
        # Details - expandable
        self.details_widget = QWidget()
        details_layout = self.create_details()
        self.details_widget.setLayout(details_layout)
        self.details_widget.hide()  # Initially collapsed
        
        layout.addWidget(self.details_widget)
        
        # Click to expand/collapse
        self.mousePressEvent = self.toggle_expanded
    
    def create_header(self):
        """Create the card header with key info"""
        layout = QHBoxLayout()
        layout.setSpacing(15)
        
        # Rank badge
        rank_label = QLabel(f"#{self.rank}")
        rank_label.setObjectName("rankBadge")
        font = QFont()
        font.setPointSize(16)
        font.setBold(True)
        rank_label.setFont(font)
        rank_label.setStyleSheet("color: #0078d4; min-width: 40px;")
        layout.addWidget(rank_label)
        
        # Strategy name and type
        name_layout = QVBoxLayout()
        name_layout.setSpacing(2)
        
        name_label = QLabel(self.strategy['name'])
        name_label.setObjectName("strategyName")
        font = QFont()
        font.setPointSize(16)
        font.setBold(True)
        name_label.setFont(font)
        name_layout.addWidget(name_label)
        
        type_label = QLabel(self.strategy['type'])
        type_label.setStyleSheet("color: #b0b0b0; font-size: 10pt;")
        name_layout.addWidget(type_label)
        
        layout.addLayout(name_layout)
        layout.addStretch()
        
        # Fit score with stars
        score_layout = QVBoxLayout()
        score_layout.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        
        score_label = QLabel(f"{self.strategy['fit_score']}/100")
        score_label.setObjectName("fitScore")
        font = QFont()
        font.setPointSize(14)
        font.setBold(True)
        score_label.setFont(font)
        score_layout.addWidget(score_label, alignment=Qt.AlignRight)
        
        # Star rating
        stars = '⭐' * min(5, int(self.strategy['fit_score'] / 20))
        stars_label = QLabel(stars)
        stars_label.setStyleSheet("font-size: 14pt;")
        score_layout.addWidget(stars_label, alignment=Qt.AlignRight)
        
        layout.addLayout(score_layout)
        
        # Expand indicator
        self.expand_indicator = QLabel("▼")
        self.expand_indicator.setStyleSheet("color: #0078d4; font-size: 12pt;")
        layout.addWidget(self.expand_indicator)
        
        return layout
    
    def create_details(self):
        """Create the expandable details section"""
        layout = QVBoxLayout()
        layout.setSpacing(15)
        
        # Separator
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setStyleSheet("background-color: #3d3d3d;")
        layout.addWidget(separator)
        
        # Description
        desc_label = QLabel(self.strategy['description'])
        desc_label.setWordWrap(True)
        desc_label.setStyleSheet("color: #e0e0e0; font-size: 11pt;")
        layout.addWidget(desc_label)
        
        # Setup steps
        setup_section = QVBoxLayout()
        setup_title = QLabel("💼 Setup:")
        setup_title.setStyleSheet("font-weight: bold; color: #0078d4; font-size: 11pt;")
        setup_section.addWidget(setup_title)
        
        for step in self.strategy['setup']:
            step_label = QLabel(f"  • {step}")
            step_label.setWordWrap(True)
            step_label.setStyleSheet("color: #e0e0e0;")
            setup_section.addWidget(step_label)
        
        layout.addLayout(setup_section)
        
        # Capital and P/L
        capital_section = QHBoxLayout()
        
        capital_label = QLabel(f"💰 Capital Required: ${self.strategy['capital_required']:,.2f}")
        capital_label.setStyleSheet("font-weight: bold; color: #00d47c;")
        capital_section.addWidget(capital_label)
        capital_section.addStretch()
        
        layout.addLayout(capital_section)
        
        # Profit/Loss box
        pl_box = QFrame()
        pl_box.setStyleSheet("""
            QFrame {
                background-color: #3d3d3d;
                border-radius: 6px;
                padding: 10px;
            }
        """)
        pl_layout = QVBoxLayout(pl_box)
        
        pl_title = QLabel("📈 Profit/Loss Potential")
        pl_title.setStyleSheet("font-weight: bold; color: #ffffff;")
        pl_layout.addWidget(pl_title)
        
        profit_label = QLabel(f"Max Profit: {self.strategy['max_profit']}")
        profit_label.setStyleSheet("color: #00d47c;")
        pl_layout.addWidget(profit_label)
        
        loss_label = QLabel(f"Max Loss: {self.strategy['max_loss']}")
        loss_label.setStyleSheet("color: #ff4444;")
        pl_layout.addWidget(loss_label)
        
        risk_label = QLabel(f"Risk Level: {self.strategy['risk_level']}")
        
        # Color code risk level
        if self.strategy['risk_level'].lower() == 'low' or 'low' in self.strategy['risk_level'].lower():
            risk_color = "#00d47c"
        elif 'medium' in self.strategy['risk_level'].lower():
            risk_color = "#ffaa00"
        else:
            risk_color = "#ff4444"
        
        risk_label.setStyleSheet(f"color: {risk_color}; font-weight: bold;")
        pl_layout.addWidget(risk_label)
        
        layout.addWidget(pl_box)
        
        # Ideal market
        market_label = QLabel(f"🎯 Ideal Market: {self.strategy['ideal_market']}")
        market_label.setStyleSheet("color: #b0b0b0; font-style: italic;")
        layout.addWidget(market_label)
        
        # Reasoning
        reasoning_section = QVBoxLayout()
        reasoning_title = QLabel("🧠 Why This Strategy?")
        reasoning_title.setStyleSheet("font-weight: bold; color: #0078d4; font-size: 11pt;")
        reasoning_section.addWidget(reasoning_title)
        
        for reason in self.strategy['reasoning']:
            reason_label = QLabel(f"  {reason}")
            reason_label.setWordWrap(True)
            reason_label.setStyleSheet("color: #e0e0e0;")
            reasoning_section.addWidget(reason_label)
        
        layout.addLayout(reasoning_section)
        
        # AI insights if available
        if 'ai_reasoning' in self.strategy and self.strategy['ai_reasoning']:
            ai_section = QVBoxLayout()
            ai_title = QLabel("🤖 AI Insight (Meta LLAMA):")
            ai_title.setStyleSheet("font-weight: bold; color: #00d47c; font-size: 11pt;")
            ai_section.addWidget(ai_title)
            
            ai_text = QLabel(self.strategy['ai_reasoning'])
            ai_text.setWordWrap(True)
            ai_text.setStyleSheet("""
                color: #e0e0e0;
                background-color: #2d2d2d;
                padding: 10px;
                border-radius: 6px;
                border-left: 4px solid #00d47c;
            """)
            ai_section.addWidget(ai_text)
            
            layout.addLayout(ai_section)
        
        # Risk assessment
        risk_section = QVBoxLayout()
        risk_status = self.strategy['risk_assessment']['status']
        risk_msg = self.strategy['risk_assessment']['message']
        
        risk_title = QLabel(f"⚖️  Risk Assessment: {risk_status}")
        risk_title.setStyleSheet("font-weight: bold; font-size: 11pt;")
        risk_section.addWidget(risk_title)
        
        risk_msg_label = QLabel(f"   {risk_msg}")
        risk_msg_label.setWordWrap(True)
        risk_msg_label.setStyleSheet("color: #b0b0b0;")
        risk_section.addWidget(risk_msg_label)
        
        layout.addLayout(risk_section)
        
        # Action buttons
        button_layout = QHBoxLayout()
        button_layout.setSpacing(10)
        
        copy_button = QPushButton("📋 Copy Details")
        copy_button.setProperty("secondary", True)
        copy_button.clicked.connect(self.copy_strategy_details)
        button_layout.addWidget(copy_button)
        
        button_layout.addStretch()
        layout.addLayout(button_layout)
        
        return layout
    
    def toggle_expanded(self, event):
        """Toggle expanded/collapsed state"""
        self.is_expanded = not self.is_expanded
        
        if self.is_expanded:
            self.details_widget.show()
            self.expand_indicator.setText("▲")
        else:
            self.details_widget.hide()
            self.expand_indicator.setText("▼")
    
    def copy_strategy_details(self):
        """Copy strategy details to clipboard"""
        text = f"""
Strategy #{self.rank}: {self.strategy['name']}
Fit Score: {self.strategy['fit_score']}/100
Type: {self.strategy['type']}

Description: {self.strategy['description']}

Setup:
"""
        for step in self.strategy['setup']:
            text += f"  • {step}\n"
        
        text += f"""
Capital Required: ${self.strategy['capital_required']:,.2f}
Max Profit: {self.strategy['max_profit']}
Max Loss: {self.strategy['max_loss']}
Risk Level: {self.strategy['risk_level']}
Ideal Market: {self.strategy['ideal_market']}

Why This Strategy:
"""
        for reason in self.strategy['reasoning']:
            text += f"  {reason}\n"
        
        text += f"""
Risk Assessment: {self.strategy['risk_assessment']['status']}
{self.strategy['risk_assessment']['message']}
"""
        
        clipboard = QApplication.clipboard()
        clipboard.setText(text)
        
        # Could show a tooltip or status message here
        self.copy_requested.emit(text)
