"""
OptionsTitan Results Widget
Tabbed interface for displaying analysis results
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTabWidget,
    QTextEdit, QLabel, QScrollArea, QGridLayout, QFrame,
    QStackedWidget
)
from PySide6.QtCore import Qt, Signal
from datetime import datetime

from .strategy_card import StrategyCard
from .charts_widget import ChartsWidget


class ResultsWidget(QTabWidget):
    """
    Tabbed widget for displaying strategy analysis results.
    Includes tabs for: Overview, Strategies, AI Insights
    """
    
    export_requested = Signal(str)  # Emits format type (pdf, html, txt)
    
    def __init__(self):
        super().__init__()
        self.setup_ui()
        self.stock_data = None
        self.strategies = None
    
    def setup_ui(self):
        """Create the tabbed interface"""
        self.setTabPosition(QTabWidget.North)
        self.setMovable(False)
        self.setDocumentMode(True)
        
        # Tab 1: Overview
        self.overview_tab = self.create_overview_tab()
        self.addTab(self.overview_tab, "📊 Overview")

        # Tab 2: Charts (dedicated tab - full width to avoid squishing)
        self.charts_tab = self.create_charts_tab()
        self.addTab(self.charts_tab, "📈 Charts")
        
        # Tab 3: Strategies
        self.strategies_tab = self.create_strategies_tab()
        self.addTab(self.strategies_tab, "🎯 Strategies")
        
        # Tab 4: AI Insights
        self.ai_tab = self.create_ai_insights_tab()
        self.addTab(self.ai_tab, "🤖 AI Insights")
        
        # Show welcome message initially
        self.show_welcome_message()
    
    def create_overview_tab(self):
        """Create the overview tab with dashboard grid layout and welcome state"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(0, 0, 0, 0)

        self.overview_stack = QStackedWidget()

        # Page 0: Dashboard welcome state
        welcome_page = self.create_welcome_page()
        self.overview_stack.addWidget(welcome_page)

        # Page 1: Results (report only - charts moved to dedicated tab)
        results_page = QWidget()
        layout_results = QVBoxLayout(results_page)
        try:
            from .ui_utils import get_responsive_spacing, get_screen_percentage_height
            spacing = get_responsive_spacing()
            text_min_h = max(220, int(get_screen_percentage_height(0.28)))
        except Exception:
            spacing = 16
            text_min_h = 280

        layout_results.setSpacing(spacing)

        report_frame = QFrame()
        report_frame.setProperty("dashboard_card", True)
        report_layout = QVBoxLayout(report_frame)
        report_layout.setContentsMargins(16, 16, 16, 16)

        self.overview_text = QTextEdit()
        self.overview_text.setReadOnly(True)
        self.overview_text.setMinimumHeight(text_min_h)
        report_layout.addWidget(self.overview_text)

        layout_results.addWidget(report_frame)
        self.overview_stack.addWidget(results_page)

        layout.addWidget(self.overview_stack)
        return tab

    def create_welcome_page(self):
        """Create dashboard-style empty state with Get Started and feature cards"""
        page = QWidget()
        layout = QVBoxLayout(page)
        try:
            from .ui_utils import get_responsive_spacing, get_screen_percentage_height
            spacing = get_responsive_spacing()
        except Exception:
            spacing = 16
        layout.setSpacing(spacing)
        layout.setContentsMargins(24, 24, 24, 24)

        # Central Get Started card
        cta_card = QFrame()
        cta_card.setProperty("dashboard_card", True)
        cta_card.setProperty("cta_card", True)
        cta_layout = QVBoxLayout(cta_card)
        cta_layout.setContentsMargins(32, 32, 32, 32)

        cta_label = QLabel("Enter a symbol and click Analyze")
        cta_label.setObjectName("ctaLabel")
        cta_label.setAlignment(Qt.AlignCenter)
        cta_layout.addWidget(cta_label)

        cta_hint = QLabel("Get personalized options strategy recommendations in seconds")
        cta_hint.setProperty("subtitle", True)
        cta_hint.setAlignment(Qt.AlignCenter)
        cta_layout.addWidget(cta_hint)

        layout.addWidget(cta_card, alignment=Qt.AlignCenter)

        # Feature cards row
        features_grid = QHBoxLayout()
        features_grid.setSpacing(spacing)

        features = [
            ("🎯", "Top 5 Strategies", "Strategies ranked by fit score"),
            ("📈", "Charts Tab", "Price history & fit scores, full width"),
            ("🤖", "AI Insights", "LLAMA-powered analysis when configured"),
        ]
        for emoji, title, desc in features:
            card = QFrame()
            card.setProperty("dashboard_card", True)
            card.setProperty("feature_card", True)
            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(20, 20, 20, 20)

            emoji_label = QLabel(emoji)
            emoji_label.setAlignment(Qt.AlignCenter)
            card_layout.addWidget(emoji_label)

            title_label = QLabel(title)
            title_label.setProperty("feature_title", True)
            title_label.setAlignment(Qt.AlignCenter)
            card_layout.addWidget(title_label)

            desc_label = QLabel(desc)
            desc_label.setProperty("feature_desc", True)
            desc_label.setAlignment(Qt.AlignCenter)
            desc_label.setWordWrap(True)
            card_layout.addWidget(desc_label)

            features_grid.addWidget(card)

        layout.addLayout(features_grid)

        layout.addStretch()
        return page

    def create_charts_tab(self):
        """Dedicated tab for charts - stacked vertically for full-width display."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(0, 0, 0, 0)
        self.charts_widget = ChartsWidget(vertical_layout=True)
        layout.addWidget(self.charts_widget)
        return tab
    
    def create_strategies_tab(self):
        """Create the strategies tab with scrollable strategy cards"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        try:
            from .ui_utils import get_responsive_spacing
            card_spacing = get_responsive_spacing()
        except Exception:
            card_spacing = 14

        self.strategies_container = QWidget()
        self.strategies_layout = QVBoxLayout(self.strategies_container)
        self.strategies_layout.setSpacing(card_spacing)
        self.strategies_layout.addStretch()

        scroll.setWidget(self.strategies_container)
        layout.addWidget(scroll)
        return tab
    
    def create_ai_insights_tab(self):
        """Create the AI insights tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        try:
            from .ui_utils import get_screen_percentage_height
            ai_min_h = max(250, int(get_screen_percentage_height(0.32)))
        except Exception:
            ai_min_h = 320

        self.ai_insights_text = QTextEdit()
        self.ai_insights_text.setReadOnly(True)
        self.ai_insights_text.setMinimumHeight(ai_min_h)
        layout.addWidget(self.ai_insights_text)
        return tab
    
    def show_welcome_message(self):
        """Display dashboard welcome state before analysis"""
        self.overview_stack.setCurrentIndex(0)
        self.overview_text.setPlainText("")
        self.ai_insights_text.setPlainText(
            "AI insights will appear here after you run an analysis.\n\n"
            "Configure LLAMA_API_KEY in .env to enable AI-powered commentary."
        )
    
    def display_results(self, stock_data, strategies):
        """Display analysis results across all tabs"""
        self.stock_data = stock_data
        self.strategies = strategies
        self.overview_stack.setCurrentIndex(1)

        # Update charts (price + fit scores)
        self.charts_widget.update_charts(stock_data, strategies)

        # Update Overview tab
        self.update_overview_tab(stock_data, strategies)
        
        # Update Strategies tab
        self.update_strategies_tab(strategies)
        
        # Update AI Insights tab
        self.update_ai_insights_tab(stock_data, strategies)
        
        # Switch to overview tab to show results
        self.setCurrentIndex(0)
    
    def update_overview_tab(self, stock_data, strategies):
        """Update the overview tab with market data"""
        
        # Calculate risk profile
        if hasattr(self, 'user_params'):
            max_risk = self.user_params.get('max_risk', 5)
            liquidity = self.user_params.get('liquidity', 10000)
            risk_profile = 'Conservative' if max_risk < 3 else 'Moderate' if max_risk < 7 else 'Aggressive'
        else:
            max_risk = 5
            liquidity = 10000
            risk_profile = 'Moderate'
        
        overview = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                        OPTIONS STRATEGY ANALYSIS REPORT                      ║
║                        Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}                       ║
╚══════════════════════════════════════════════════════════════════════════════╝

📊 MARKET OVERVIEW - {stock_data['symbol']}
{'=' * 80}
Current Price:      ${stock_data['current_price']:.2f}
52-Week High:       ${stock_data['high_52w']:.2f}
52-Week Low:        ${stock_data['low_52w']:.2f}
Implied Volatility: {stock_data['volatility']:.1%}
Market Trend:       {stock_data['trend']}

💰 YOUR PARAMETERS
{'=' * 80}
Portfolio Liquidity: ${liquidity:,.2f}
Max Risk per Trade:  {max_risk:.1f}% (${liquidity * max_risk / 100:,.2f})
Risk Profile:        {risk_profile}

"""
        
        # Add AI market insights if available
        if strategies and 'market_insights' in strategies[0] and strategies[0]['market_insights']:
            overview += f"""
🤖 AI MARKET INSIGHTS (Powered by Meta LLAMA)
{'=' * 80}
{strategies[0]['market_insights']}

"""
        
        overview += f"""
🎯 TOP 5 RECOMMENDED STRATEGIES
{'=' * 80}

"""
        
        # Add strategy summaries
        for i, strategy in enumerate(strategies, 1):
            stars = '⭐' * min(5, int(strategy['fit_score'] / 20))
            overview += f"""
{i}. {strategy['name']} {stars}
   Fit Score: {strategy['fit_score']}/100
   Type: {strategy['type']}
   Risk Level: {strategy['risk_level']}
   Capital Required: ${strategy['capital_required']:,.2f}

"""
        
        # Add AI risk commentary if available
        if strategies and 'risk_commentary' in strategies[0] and strategies[0]['risk_commentary']:
            overview += f"""
🤖 AI RISK ASSESSMENT (Powered by Meta LLAMA)
{'=' * 80}
{strategies[0]['risk_commentary']}

"""
        
        overview += """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

⚠️  IMPORTANT REMINDERS:
   • Start with paper trading to practice without real money
   • Never risk more than you can afford to lose
   • Options are complex instruments - understand them before trading
   • Market conditions can change rapidly - monitor your positions
   • These are educational recommendations, not financial advice

💡 NEXT STEPS:
   1. Review each strategy's risk/reward profile in the Strategies tab
   2. Check current options chain for exact pricing
   3. Consider your market outlook and time horizon
   4. Start with the highest-scored strategy that fits your risk tolerance
   5. Always set stop-loss orders to protect your capital

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        
        self.overview_text.setPlainText(overview)
    
    def update_strategies_tab(self, strategies):
        """Update the strategies tab with strategy cards"""
        
        # Clear existing cards
        while self.strategies_layout.count() > 1:  # Keep the stretch
            item = self.strategies_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        # Add strategy cards
        for i, strategy in enumerate(strategies, 1):
            card = StrategyCard(strategy, i)
            card.copy_requested.connect(lambda text: self.show_copy_confirmation())
            self.strategies_layout.insertWidget(i-1, card)
    
    def update_ai_insights_tab(self, stock_data, strategies):
        """Update the AI insights tab"""
        
        ai_text = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                           AI-POWERED INSIGHTS                                ║
║                        Powered by Meta LLAMA AI                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

"""
        
        # Market insights
        if strategies and 'market_insights' in strategies[0] and strategies[0]['market_insights']:
            ai_text += f"""
📊 MARKET ANALYSIS for {stock_data['symbol']}
{'=' * 80}

{strategies[0]['market_insights']}

"""
        else:
            ai_text += """
📊 MARKET ANALYSIS
{'=' * 80}

AI market insights not available. Ensure LLAMA_API_KEY is configured.

"""
        
        # Strategy insights (top 3)
        ai_text += """
💡 STRATEGY-SPECIFIC INSIGHTS
{'=' * 80}

"""
        
        for i, strategy in enumerate(strategies[:3], 1):
            if 'ai_reasoning' in strategy and strategy['ai_reasoning']:
                ai_text += f"""
{i}. {strategy['name']}
───────────────────────────────────────────────────────────────────────────────
{strategy['ai_reasoning']}

"""
            else:
                # Fallback: use built-in reasoning when LLAMA AI is not available
                built_in = strategy.get('reasoning', [])
                if built_in:
                    reasoning_text = "\n".join(f"  • {line}" for line in built_in)
                    ai_text += f"""
{i}. {strategy['name']}
───────────────────────────────────────────────────────────────────────────────
Built-in analysis (configure LLAMA_API_KEY for AI-powered insights):

{reasoning_text}

"""
                else:
                    ai_text += f"""
{i}. {strategy['name']}
───────────────────────────────────────────────────────────────────────────────
Configure LLAMA_API_KEY in .env for AI-powered reasoning. See LLAMA_AI_SETUP.md.

"""
        
        # Risk assessment
        if strategies and 'risk_commentary' in strategies[0] and strategies[0]['risk_commentary']:
            ai_text += f"""
⚖️  RISK MANAGEMENT ASSESSMENT
{'=' * 80}

{strategies[0]['risk_commentary']}

"""
        
        ai_text += """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✨ These insights are generated by Meta's LLAMA AI to provide personalized
   analysis based on your specific parameters and current market conditions.

⚠️  Remember: AI insights are educational tools. Always do your own research
    and consult with financial professionals before making trading decisions.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        
        self.ai_insights_text.setPlainText(ai_text)
    
    def show_copy_confirmation(self):
        """Show confirmation that text was copied"""
        # Could implement a temporary status message here
        pass
    
    def clear_results(self):
        """Clear all results and show welcome message"""
        self.stock_data = None
        self.strategies = None
        self.show_welcome_message()
        self.charts_widget.update_charts(None, [])

        # Clear strategy cards
        while self.strategies_layout.count() > 1:
            item = self.strategies_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
    
    def set_user_params(self, liquidity, max_risk, target_profit, max_loss):
        """Store user parameters for display"""
        self.user_params = {
            'liquidity': liquidity,
            'max_risk': max_risk,
            'target_profit': target_profit,
            'max_loss': max_loss
        }
