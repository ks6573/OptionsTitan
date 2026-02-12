"""
OptionsTitan Strategy Card Widget
Responsive expandable strategy cards
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QFrame, QApplication
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont


def _base_font_size():
    """Responsive base font size in pixels."""
    try:
        from .ui_utils import get_responsive_font_size
        return get_responsive_font_size('body')
    except Exception:
        return 13


class StrategyCard(QFrame):
    """
    A responsive card widget displaying a single strategy's details.
    Can be expanded/collapsed to show full information.
    """

    copy_requested = Signal(str)

    def __init__(self, strategy, rank, parent=None):
        super().__init__(parent)
        self.strategy = strategy
        self.rank = rank
        self.is_expanded = False
        self._base = _base_font_size()
        self.setup_ui()

    def setup_ui(self):
        self.setObjectName("strategyCard")
        self.setFrameShape(QFrame.StyledPanel)
        self.setFrameShadow(QFrame.Raised)
        self.setCursor(Qt.PointingHandCursor)

        layout = QVBoxLayout(self)
        try:
            from .ui_utils import get_responsive_spacing
            layout.setSpacing(get_responsive_spacing())
        except Exception:
            layout.setSpacing(10)

        header_layout = self.create_header()
        layout.addLayout(header_layout)

        self.details_widget = QWidget()
        details_layout = self.create_details()
        self.details_widget.setLayout(details_layout)
        self.details_widget.hide()
        layout.addWidget(self.details_widget)
        self.mousePressEvent = self.toggle_expanded

    def create_header(self):
        layout = QHBoxLayout()
        layout.setSpacing(max(10, int(self._base * 1.2)))

        rank_label = QLabel(f"#{self.rank}")
        rank_label.setObjectName("rankBadge")
        font = QFont()
        font.setPixelSize(int(self._base * 1.4))
        font.setBold(True)
        rank_label.setFont(font)
        rank_label.setStyleSheet("color: #0078d4; min-width: 32px;")
        layout.addWidget(rank_label)

        name_layout = QVBoxLayout()
        name_layout.setSpacing(2)

        name_label = QLabel(self.strategy['name'])
        name_label.setObjectName("strategyName")
        font = QFont()
        font.setPixelSize(int(self._base * 1.3))
        font.setBold(True)
        name_label.setFont(font)
        name_layout.addWidget(name_label)

        type_label = QLabel(self.strategy['type'])
        type_label.setStyleSheet(f"color: #b0b0b0; font-size: {int(self._base * 0.9)}px;")
        name_layout.addWidget(type_label)
        layout.addLayout(name_layout)
        layout.addStretch()

        score_layout = QVBoxLayout()
        score_layout.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        score_label = QLabel(f"{self.strategy['fit_score']}/100")
        score_label.setObjectName("fitScore")
        font = QFont()
        font.setPixelSize(int(self._base * 1.25))
        font.setBold(True)
        score_label.setFont(font)
        score_layout.addWidget(score_label, alignment=Qt.AlignRight)

        stars = '⭐' * min(5, int(self.strategy['fit_score'] / 20))
        stars_label = QLabel(stars)
        stars_label.setStyleSheet(f"font-size: {int(self._base * 1.15)}px;")
        score_layout.addWidget(stars_label, alignment=Qt.AlignRight)
        layout.addLayout(score_layout)

        self.expand_indicator = QLabel("▼")
        self.expand_indicator.setStyleSheet(f"color: #0078d4; font-size: {int(self._base * 1.0)}px;")
        layout.addWidget(self.expand_indicator)
        return layout

    def create_details(self):
        layout = QVBoxLayout()
        try:
            from .ui_utils import get_responsive_spacing
            layout.setSpacing(get_responsive_spacing())
        except Exception:
            layout.setSpacing(12)

        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setStyleSheet("background-color: #3d3d3d;")
        layout.addWidget(separator)

        desc_label = QLabel(self.strategy['description'])
        desc_label.setWordWrap(True)
        desc_label.setStyleSheet(f"color: #e0e0e0; font-size: {int(self._base * 0.95)}px;")
        layout.addWidget(desc_label)

        setup_section = QVBoxLayout()
        setup_title = QLabel("💼 Setup:")
        setup_title.setStyleSheet(f"font-weight: bold; color: #0078d4; font-size: {int(self._base * 0.95)}px;")
        setup_section.addWidget(setup_title)

        for step in self.strategy['setup']:
            step_label = QLabel(f"  • {step}")
            step_label.setWordWrap(True)
            step_label.setStyleSheet(f"color: #e0e0e0; font-size: {int(self._base * 0.9)}px;")
            setup_section.addWidget(step_label)
        layout.addLayout(setup_section)

        capital_label = QLabel(f"💰 Capital Required: ${self.strategy['capital_required']:,.2f}")
        capital_label.setStyleSheet("font-weight: bold; color: #00d47c;")
        cap_layout = QHBoxLayout()
        cap_layout.addWidget(capital_label)
        cap_layout.addStretch()
        layout.addLayout(cap_layout)

        pl_box = QFrame()
        pl_box.setStyleSheet("""
            QFrame {
                background-color: #3d3d3d;
                border-radius: 6px;
                padding: 8px;
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
        if self.strategy['risk_level'].lower() == 'low' or 'low' in self.strategy['risk_level'].lower():
            risk_color = "#00d47c"
        elif 'medium' in self.strategy['risk_level'].lower():
            risk_color = "#ffaa00"
        else:
            risk_color = "#ff4444"
        risk_label.setStyleSheet(f"color: {risk_color}; font-weight: bold;")
        pl_layout.addWidget(risk_label)
        layout.addWidget(pl_box)

        market_label = QLabel(f"🎯 Ideal Market: {self.strategy['ideal_market']}")
        market_label.setStyleSheet("color: #b0b0b0; font-style: italic;")
        layout.addWidget(market_label)

        reasoning_section = QVBoxLayout()
        reasoning_title = QLabel("🧠 Why This Strategy?")
        reasoning_title.setStyleSheet(f"font-weight: bold; color: #0078d4; font-size: {int(self._base * 0.95)}px;")
        reasoning_section.addWidget(reasoning_title)
        for reason in self.strategy['reasoning']:
            reason_label = QLabel(f"  {reason}")
            reason_label.setWordWrap(True)
            reason_label.setStyleSheet(f"color: #e0e0e0; font-size: {int(self._base * 0.9)}px;")
            reasoning_section.addWidget(reason_label)
        layout.addLayout(reasoning_section)

        if 'ai_reasoning' in self.strategy and self.strategy['ai_reasoning']:
            ai_section = QVBoxLayout()
            ai_title = QLabel("🤖 AI Insight (Meta LLAMA):")
            ai_title.setStyleSheet(f"font-weight: bold; color: #00d47c; font-size: {int(self._base * 0.95)}px;")
            ai_section.addWidget(ai_title)
            ai_text = QLabel(self.strategy['ai_reasoning'])
            ai_text.setWordWrap(True)
            ai_text.setStyleSheet("""
                color: #e0e0e0;
                background-color: #2d2d2d;
                padding: 8px;
                border-radius: 6px;
                border-left: 4px solid #00d47c;
            """)
            ai_section.addWidget(ai_text)
            layout.addLayout(ai_section)

        risk_section = QVBoxLayout()
        risk_status = self.strategy['risk_assessment']['status']
        risk_msg = self.strategy['risk_assessment']['message']
        risk_title = QLabel(f"⚖️  Risk Assessment: {risk_status}")
        risk_title.setStyleSheet(f"font-weight: bold; font-size: {int(self._base * 0.95)}px;")
        risk_section.addWidget(risk_title)
        risk_msg_label = QLabel(f"   {risk_msg}")
        risk_msg_label.setWordWrap(True)
        risk_msg_label.setStyleSheet("color: #b0b0b0;")
        risk_section.addWidget(risk_msg_label)
        layout.addLayout(risk_section)

        button_layout = QHBoxLayout()
        copy_button = QPushButton("📋 Copy Details")
        copy_button.setProperty("secondary", True)
        copy_button.clicked.connect(self.copy_strategy_details)
        button_layout.addWidget(copy_button)
        button_layout.addStretch()
        layout.addLayout(button_layout)
        return layout

    def toggle_expanded(self, event):
        self.is_expanded = not self.is_expanded
        if self.is_expanded:
            self.details_widget.show()
            self.expand_indicator.setText("▲")
        else:
            self.details_widget.hide()
            self.expand_indicator.setText("▼")

    def copy_strategy_details(self):
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
        QApplication.clipboard().setText(text)
        self.copy_requested.emit(text)
