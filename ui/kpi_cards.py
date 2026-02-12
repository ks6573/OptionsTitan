"""
OptionsTitan KPI Cards
Dashboard-style stat cards for at-a-glance metrics
"""

from PySide6.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QLabel, QFrame
from PySide6.QtCore import Qt


class KPICard(QFrame):
    """Single KPI card: label + value."""

    def __init__(self, label: str, value: str = "—", parent=None):
        super().__init__(parent)
        self.setProperty("kpi_card", True)
        self.setObjectName("kpiCard")

        layout = QVBoxLayout(self)
        layout.setSpacing(4)
        layout.setContentsMargins(16, 14, 16, 14)

        self.label_widget = QLabel(label)
        self.label_widget.setStyleSheet("color: #94a3b8; font-size: 13px;")
        layout.addWidget(self.label_widget)

        self.value_widget = QLabel(value)
        self.value_widget.setStyleSheet("color: #ffffff; font-size: 22px; font-weight: bold;")
        layout.addWidget(self.value_widget)

    def set_value(self, value: str):
        self.value_widget.setText(value)


class KPIStrip(QWidget):
    """Horizontal row of KPI cards. Hidden when empty, visible when set_data() is called."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setProperty("kpi_strip", True)
        layout = QHBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(0, 0, 0, 0)

        self.cards = []
        for label in ("Symbol", "Strategies", "Top Score", "Trend"):
            card = KPICard(label, "—")
            self.cards.append(card)
            layout.addWidget(card)

        self.hide()  # Hidden until we have data

    def set_data(self, stock_data: dict, strategies: list):
        """Populate KPI cards from analysis results."""
        if not stock_data or not strategies:
            self.clear_data()
            return

        symbol = stock_data.get("symbol", "—")
        count = len(strategies)
        top_score = strategies[0]["fit_score"] if strategies else "—"
        trend = stock_data.get("trend", "—")

        self.cards[0].set_value(str(symbol))
        self.cards[1].set_value(str(count))
        self.cards[2].set_value(str(top_score) if top_score != "—" else "—")
        self.cards[3].set_value(str(trend))

        self.show()

    def clear_data(self):
        """Reset and hide KPI strip."""
        for card in self.cards:
            card.set_value("—")
        self.hide()
