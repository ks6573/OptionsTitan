"""
OptionsTitan Charts Widget
Responsive price history and strategy fit score visualizations
With tooltips and improved visibility
"""

import pandas as pd
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSizePolicy, QToolTip
from PySide6.QtCharts import (
    QChart, QChartView, QLineSeries, QValueAxis, QDateTimeAxis,
    QHorizontalBarSeries, QBarSet, QBarCategoryAxis
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QPainter, QColor, QFont

from .ui_utils import get_screen_percentage_height


def _get_chart_heights():
    """Responsive chart min/max heights based on screen."""
    screen_h = get_screen_percentage_height(1.0)
    min_h = max(150, int(screen_h * 0.18))
    max_h = int(screen_h * 0.38)
    return min_h, max_h


def build_price_chart(price_history, symbol: str) -> QChartView:
    """Build a responsive 60-day price chart with tooltips and zoom."""
    min_h, max_h = _get_chart_heights()

    chart = QChart()
    chart.setTitle(f"{symbol} — 60-Day Price")
    chart.setAnimationOptions(QChart.SeriesAnimations)
    chart.setBackgroundBrush(QColor("#1e1e1e"))
    chart.setTitleBrush(QColor("#ffffff"))

    title_font = QFont()
    title_font.setPixelSize(max(10, int(min_h / 18)))
    title_font.setBold(True)
    chart.setTitleFont(title_font)

    series = QLineSeries()
    series.setColor(QColor("#0d9bff"))
    pen = series.pen()
    pen.setWidth(2)
    series.setPen(pen)

    if price_history is not None and not price_history.empty:
        for dt, row in price_history.iterrows():
            ts = int(pd.Timestamp(dt).timestamp() * 1000)
            series.append(ts, float(row['Close']))

    chart.addSeries(series)

    axis_x = QDateTimeAxis()
    axis_x.setFormat("MMM d")
    axis_x.setLabelsColor(QColor("#b0b0b0"))
    axis_font = QFont()
    axis_font.setPixelSize(max(8, int(min_h / 22)))
    axis_x.setLabelsFont(axis_font)
    chart.addAxis(axis_x, Qt.AlignBottom)
    series.attachAxis(axis_x)

    axis_y = QValueAxis()
    axis_y.setLabelsColor(QColor("#b0b0b0"))
    axis_y.setLabelsFont(axis_font)
    chart.addAxis(axis_y, Qt.AlignLeft)
    series.attachAxis(axis_y)

    view = QChartView(chart)
    view.setRenderHint(QPainter.Antialiasing)
    view.setBackgroundBrush(QColor("#1e1e1e"))
    view.setMinimumHeight(min_h)
    view.setMaximumHeight(max_h)
    view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)

    # Enable rubber band zoom (click-drag to zoom)
    try:
        view.setRubberBand(QChartView.RectangleRubberBand)
    except Exception:
        pass

    # Tooltip on hover
    def on_hovered(point, state):
        if state and point:
            ts_ms = int(point.x())
            dt = pd.Timestamp(ts_ms / 1000.0).strftime("%b %d, %Y")
            price = point.y()
            from PySide6.QtWidgets import QApplication
            QToolTip.showText(QApplication.instance().cursor().pos(), f"{dt}\n${price:,.2f}")
        else:
            QToolTip.hideText()

    series.hovered.connect(on_hovered)

    return view


def build_fit_score_chart(strategies: list) -> QChartView:
    """Build a horizontal bar chart with visible bars and tooltips."""
    min_h, max_h = _get_chart_heights()

    chart = QChart()
    chart.setTitle("Strategy Fit Scores")
    chart.setAnimationOptions(QChart.SeriesAnimations)
    chart.setBackgroundBrush(QColor("#1e1e1e"))
    chart.setTitleBrush(QColor("#ffffff"))

    title_font = QFont()
    title_font.setPixelSize(max(10, int(min_h / 18)))
    title_font.setBold(True)
    chart.setTitleFont(title_font)

    bar_set = QBarSet("Fit Score")
    bar_set.setColor(QColor("#0d9bff"))
    bar_set.setBorderColor(QColor("#4da6ff"))
    bar_set.setLabelColor(QColor("#ffffff"))

    categories = []
    for s in strategies:
        bar_set.append(float(s['fit_score']))
        categories.append(s['name'].replace(" ", "\n"))

    # QHorizontalBarSeries: categories on Y (left), values extend right
    series = QHorizontalBarSeries()
    series.setBarWidth(0.65)
    series.append(bar_set)
    series.setLabelsVisible(True)
    series.setLabelsFormat("@value")
    try:
        from PySide6.QtCharts import QAbstractBarSeries
        series.setLabelsPosition(QAbstractBarSeries.LabelsOutsideEnd)
        series.setLabelsAngle(0)
    except Exception:
        pass
    chart.addSeries(series)

    axis_y = QBarCategoryAxis()
    axis_y.append(categories)
    axis_y.setLabelsColor(QColor("#b0b0b0"))
    axis_font = QFont()
    axis_font.setPixelSize(max(9, int(min_h / 20)))
    axis_y.setLabelsFont(axis_font)
    chart.addAxis(axis_y, Qt.AlignLeft)
    series.attachAxis(axis_y)

    axis_x = QValueAxis()
    axis_x.setRange(0, 100)
    axis_x.setLabelsColor(QColor("#b0b0b0"))
    axis_x.setLabelsFont(axis_font)
    axis_x.setGridLineVisible(True)
    axis_x.setMinorGridLineVisible(False)
    chart.addAxis(axis_x, Qt.AlignBottom)
    series.attachAxis(axis_x)

    chart.legend().setVisible(False)
    chart.setMargins(chart.margins())

    view = QChartView(chart)
    view.setRenderHint(QPainter.Antialiasing)
    view.setRenderHint(QPainter.SmoothPixmapTransform)
    view.setBackgroundBrush(QColor("#1e1e1e"))
    view.setMinimumHeight(min_h)
    view.setMaximumHeight(max_h)
    view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)

    # Tooltip on bar hover - hovered(state, index, barset)
    def on_bar_hovered(state, index, barset):
        if state and 0 <= index < len(strategies):
            s = strategies[index]
            tip = f"{s['name']}: {s['fit_score']}/100\n{s['type']}\nRisk: {s['risk_level']}"
            from PySide6.QtWidgets import QApplication
            QToolTip.showText(QApplication.instance().cursor().pos(), tip)
        else:
            QToolTip.hideText()

    series.hovered.connect(on_bar_hovered)

    return view


class ChartsWidget(QWidget):
    """Container for price and strategy fit charts."""

    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout(self)
        try:
            from .ui_utils import get_responsive_spacing
            self.layout.setSpacing(get_responsive_spacing())
        except Exception:
            self.layout.setSpacing(16)
        self.price_chart = None
        self.fit_chart = None

    def update_charts(self, stock_data, strategies):
        """Rebuild charts from analysis results."""
        def _spacing():
            try:
                from .ui_utils import get_responsive_spacing
                return get_responsive_spacing()
            except Exception:
                return 16

        while self.layout.count():
            item = self.layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        if not stock_data and not strategies:
            min_h = max(140, int(get_screen_percentage_height(0.20)))
            placeholder = QLabel("📈 Charts will appear here after analysis")
            placeholder.setAlignment(Qt.AlignCenter)
            fs = max(11, int(min_h / 12))
            placeholder.setStyleSheet(f"color: #666; font-size: {fs}px; padding: 24px;")
            placeholder.setMinimumHeight(min_h)
            self.layout.addWidget(placeholder)
            return

        charts_layout = QHBoxLayout()
        charts_layout.setSpacing(_spacing())

        price_history = stock_data.get('price_history') if stock_data else None
        min_h, _ = _get_chart_heights()

        if price_history is not None and not price_history.empty:
            self.price_chart = build_price_chart(price_history, stock_data['symbol'])
            charts_layout.addWidget(self.price_chart, 3)
        else:
            ph = QLabel("Price chart unavailable")
            ph.setAlignment(Qt.AlignCenter)
            ph.setStyleSheet(f"color: #666; font-size: {max(10, int(min_h/18))}px;")
            ph.setMinimumHeight(min_h)
            charts_layout.addWidget(ph, 3)

        if strategies:
            self.fit_chart = build_fit_score_chart(strategies)
            charts_layout.addWidget(self.fit_chart, 2)
        else:
            ph2 = QLabel("Strategy fit scores")
            ph2.setAlignment(Qt.AlignCenter)
            ph2.setStyleSheet(f"color: #666; font-size: {max(10, int(min_h/18))}px;")
            ph2.setMinimumHeight(min_h)
            charts_layout.addWidget(ph2, 2)

        self.layout.addLayout(charts_layout)
