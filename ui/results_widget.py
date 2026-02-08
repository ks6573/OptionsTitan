"""
OptionsTitan Results Widget
Tabbed interface for displaying analysis results
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTabWidget, 
    QTextEdit, QLabel, QScrollArea, QPushButton, QFrame
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont, QTextCharFormat, QColor
from datetime import datetime
from .strategy_card import StrategyCard


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
        
        # Tab 2: Strategies
        self.strategies_tab = self.create_strategies_tab()
        self.addTab(self.strategies_tab, "🎯 Strategies")
        
        # Tab 3: AI Insights
        self.ai_tab = self.create_ai_insights_tab()
        self.addTab(self.ai_tab, "🤖 AI Insights")
        
        # Show welcome message initially
        self.show_welcome_message()
    
    def create_overview_tab(self):
        """Create the overview tab with market data"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Text display (export buttons now in main header)
        self.overview_text = QTextEdit()
        self.overview_text.setReadOnly(True)
        self.overview_text.setMinimumHeight(400)
        layout.addWidget(self.overview_text)
        
        return tab
    
    def create_strategies_tab(self):
        """Create the strategies tab with expandable cards"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Scroll area for strategy cards
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        self.strategies_container = QWidget()
        self.strategies_layout = QVBoxLayout(self.strategies_container)
        self.strategies_layout.setSpacing(15)
        self.strategies_layout.addStretch()
        
        scroll.setWidget(self.strategies_container)
        layout.addWidget(scroll)
        
        return tab
    
    def create_ai_insights_tab(self):
        """Create the AI insights tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # AI insights text
        self.ai_insights_text = QTextEdit()
        self.ai_insights_text.setReadOnly(True)
        self.ai_insights_text.setMinimumHeight(400)
        layout.addWidget(self.ai_insights_text)
        
        return tab
    
    def show_welcome_message(self):
        """Display welcome message before analysis"""
        welcome = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                   Welcome to OptionsTitan Strategy Analyzer                  ║
║                     AI-Powered Options Strategy Analysis                     ║
╚══════════════════════════════════════════════════════════════════════════════╝


WHAT THIS TOOL DOES:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

This professional-grade analyzer helps you identify optimal options strategies based on:

  • Your risk tolerance and portfolio size
  • Current market conditions and volatility
  • Stock-specific technical and fundamental analysis
  • AI-generated market insights (when LLAMA AI is enabled)


HOW TO USE:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  1. Enter your stock symbol (e.g., SPY, AAPL, TSLA, QQQ)
  
  2. Set your portfolio liquidity (total available trading capital)
  
  3. Define your risk parameters:
     - Max Risk %: Maximum percentage of portfolio to risk per trade
     - Target Profit %: Your desired profit target
     - Max Loss %: Maximum acceptable loss threshold
  
  4. Click "Analyze Strategies" to receive personalized recommendations


WHAT YOU'LL GET:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  ✓ Top 5 strategies ranked by fit score
  ✓ Detailed setup instructions for each strategy
  ✓ Comprehensive risk/reward analysis
  ✓ Market condition alignment assessment
  ✓ Clear reasoning for each recommendation
  ✓ AI-powered insights and market commentary (with LLAMA AI)


⚠️  IMPORTANT DISCLAIMER:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    This tool is for EDUCATIONAL PURPOSES ONLY. Options trading involves
    substantial risk of loss and is not suitable for all investors.
    
    Past performance does not guarantee future results. Always conduct your
    own research and consult with a licensed financial advisor before making
    any investment decisions.
    
    By using this tool, you acknowledge that you understand these risks and
    accept full responsibility for your trading decisions.


Ready to begin? Enter your parameters in the left panel and click "Analyze Strategies"!
"""
        
        self.overview_text.setPlainText(welcome)
        self.ai_insights_text.setPlainText(
            "AI Insights will appear here after you run an analysis.\n\n"
            "To enable LLAMA AI-powered market commentary, ensure your LLAMA_API_KEY\n"
            "is properly configured in the .env file."
        )
    
    def display_results(self, stock_data, strategies):
        """Display analysis results across all tabs"""
        self.stock_data = stock_data
        self.strategies = strategies
        
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
                ai_text += f"""
{i}. {strategy['name']}
───────────────────────────────────────────────────────────────────────────────
AI reasoning not available for this strategy.

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
        self.show_welcome_message()
        
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
