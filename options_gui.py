#!/usr/bin/env python3
"""
OptionsTitan - Interactive GUI for Options Strategy Analysis
A user-friendly interface to analyze options strategies based on risk tolerance and goals
Enhanced with Meta LLAMA AI for intelligent insights
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import sys
import os
from datetime import datetime, timedelta
import threading
import json

# Load environment variables from .env file (if exists)
try:
    from dotenv import load_dotenv
    load_dotenv()  # Load .env file automatically
    print("✅ Loaded environment from .env file")
except ImportError:
    # If python-dotenv not installed, try to load .env manually
    env_file = os.path.join(os.path.dirname(__file__), '.env')
    if os.path.exists(env_file):
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    # Remove quotes if present
                    value = value.strip().strip('"').strip("'")
                    os.environ[key.strip()] = value
        print("✅ Loaded environment from .env file (manual)")

# Add src directory to Python path
src_path = os.path.join(os.path.dirname(__file__), 'src')
sys.path.insert(0, src_path)

try:
    import yfinance as yf
    import pandas as pd
    import numpy as np
    from src.risk_management import AdvancedRiskManager
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    DEPENDENCIES_AVAILABLE = False
    print(f"Warning: Some dependencies not available: {e}")

# LLAMA API Integration
try:
    from llama_api_client import LlamaAPIClient
    LLAMA_AVAILABLE = True
except ImportError:
    LLAMA_AVAILABLE = False
    print("Info: LLAMA API client not available. Install with: pip install llama-api-client")


class LLAMAEnhancer:
    """Enhances strategy analysis with Meta LLAMA AI insights"""
    
    def __init__(self):
        self.client = None
        self.enabled = False
        
        if LLAMA_AVAILABLE:
            try:
                api_key = os.environ.get("LLAMA_API_KEY")
                if api_key:
                    self.client = LlamaAPIClient(
                        api_key=api_key,
                        base_url="https://api.llama.com/v1/",
                    )
                    self.enabled = True
                    print("✅ LLAMA AI Enhancement: ENABLED")
                else:
                    print("⚠️  LLAMA_API_KEY not found in environment variables")
            except Exception as e:
                print(f"⚠️  LLAMA API initialization failed: {e}")
                self.enabled = False
        else:
            print("ℹ️  LLAMA API client not installed (optional feature)")
    
    def get_market_insights(self, stock_data, user_params):
        """Get AI-powered market insights"""
        if not self.enabled:
            return None
        
        try:
            prompt = f"""As an expert options trading analyst, provide a concise market analysis for {stock_data['symbol']}.

Current Market Data:
- Price: ${stock_data['current_price']:.2f}
- Volatility: {stock_data['volatility']:.1%}
- Trend: {stock_data['trend']}
- 52W High: ${stock_data['high_52w']:.2f}
- 52W Low: ${stock_data['low_52w']:.2f}

Trader Profile:
- Available Capital: ${user_params['liquidity']:,.0f}
- Risk Tolerance: {user_params['max_risk_pct']}% per trade
- Target Profit: {user_params['target_profit']}%

Provide a brief (3-4 sentences) analysis covering:
1. Current market conditions for this stock
2. Key opportunities or risks to consider
3. General outlook for options trading

Keep it practical and actionable."""

            response = self.client.chat.completions.create(
                model="Llama-4-Maverick-17B-128E-Instruct-FP8",
                messages=[
                    {"role": "user", "content": prompt},
                ],
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"LLAMA API Error (market insights): {e}")
            return None
    
    def enhance_strategy_reasoning(self, strategy, stock_data, user_params):
        """Enhance strategy explanation with AI insights"""
        if not self.enabled:
            return None
        
        try:
            prompt = f"""As an options trading expert, explain why the {strategy['name']} strategy is suitable for {stock_data['symbol']}.

Strategy: {strategy['name']}
Type: {strategy['type']}
Fit Score: {strategy['fit_score']}/100

Market Context:
- Price: ${stock_data['current_price']:.2f}
- Volatility: {stock_data['volatility']:.1%}
- Trend: {stock_data['trend']}

Trader Context:
- Capital: ${user_params['liquidity']:,.0f}
- Risk Tolerance: {user_params['max_risk_pct']}%
- Goal: {user_params['target_profit']}% profit

In 2-3 sentences, explain:
1. Why this strategy fits the current market environment
2. What specific advantage it offers for this trader's goals

Be concise and practical."""

            response = self.client.chat.completions.create(
                model="Llama-4-Maverick-17B-128E-Instruct-FP8",
                messages=[
                    {"role": "user", "content": prompt},
                ],
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"LLAMA API Error (strategy reasoning): {e}")
            return None
    
    def get_risk_assessment_commentary(self, strategies, user_params):
        """Get AI commentary on risk profile"""
        if not self.enabled:
            return None
        
        try:
            prompt = f"""As a risk management expert, provide a brief assessment of this trader's options strategy approach.

Trader Profile:
- Portfolio Size: ${user_params['liquidity']:,.0f}
- Max Risk per Trade: {user_params['max_risk_pct']}%
- Target Profit: {user_params['target_profit']}%
- Max Loss Tolerance: {user_params['max_loss']}%

Top Recommended Strategy: {strategies[0]['name']} (Fit: {strategies[0]['fit_score']}/100)
Risk Level: {strategies[0]['risk_level']}

In 2-3 sentences:
1. Assess if this risk profile is appropriate for options trading
2. Provide one key recommendation for managing risk

Be supportive but realistic."""

            response = self.client.chat.completions.create(
                model="Llama-4-Maverick-17B-128E-Instruct-FP8",
                messages=[
                    {"role": "user", "content": prompt},
                ],
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"LLAMA API Error (risk assessment): {e}")
            return None
    
    def compare_strategies(self, strategy1, strategy2, stock_data):
        """Get AI comparison between two strategies"""
        if not self.enabled:
            return None
        
        try:
            prompt = f"""Compare these two options strategies for {stock_data['symbol']} (currently ${stock_data['current_price']:.2f}, {stock_data['volatility']:.1%} volatility):

Strategy A: {strategy1['name']}
- Fit Score: {strategy1['fit_score']}/100
- Type: {strategy1['type']}
- Risk Level: {strategy1['risk_level']}

Strategy B: {strategy2['name']}
- Fit Score: {strategy2['fit_score']}/100
- Type: {strategy2['type']}
- Risk Level: {strategy2['risk_level']}

In 2-3 sentences, explain the key trade-off between these strategies."""

            response = self.client.chat.completions.create(
                model="Llama-4-Maverick-17B-128E-Instruct-FP8",
                messages=[
                    {"role": "user", "content": prompt},
                ],
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"LLAMA API Error (strategy comparison): {e}")
            return None


class OptionsStrategyAnalyzer:
    """Analyzes and generates options strategy recommendations"""
    
    def __init__(self):
        self.risk_manager = AdvancedRiskManager()
        self.llama_enhancer = LLAMAEnhancer()
        
    def get_stock_data(self, symbol, days=60):
        """Fetch stock data from Yahoo Finance"""
        try:
            ticker = yf.Ticker(symbol)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # Get historical data
            hist = ticker.history(start=start_date, end=end_date)
            
            # Get current price
            current_price = hist['Close'].iloc[-1] if not hist.empty else None
            
            # Calculate volatility (annualized)
            returns = hist['Close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252) if len(returns) > 0 else 0.30
            
            # Calculate trend
            sma_20 = hist['Close'].rolling(window=20).mean()
            trend = "Bullish" if hist['Close'].iloc[-1] > sma_20.iloc[-1] else "Bearish"
            
            return {
                'symbol': symbol,
                'current_price': current_price,
                'volatility': volatility,
                'trend': trend,
                'returns': returns,
                'high_52w': hist['High'].max(),
                'low_52w': hist['Low'].min()
            }
        except Exception as e:
            raise Exception(f"Error fetching data for {symbol}: {str(e)}")
    
    def generate_strategies(self, stock_data, liquidity, max_risk_pct, target_profit, max_loss):
        """Generate top 5 options strategies based on user inputs"""
        
        symbol = stock_data['symbol']
        price = stock_data['current_price']
        volatility = stock_data['volatility']
        trend = stock_data['trend']
        
        # Calculate position size based on risk
        max_risk_dollar = liquidity * (max_risk_pct / 100)
        
        # User parameters for LLAMA enhancement
        user_params = {
            'liquidity': liquidity,
            'max_risk_pct': max_risk_pct,
            'target_profit': target_profit,
            'max_loss': max_loss
        }
        
        strategies = []
        
        # Strategy 1: Covered Call (Conservative Income)
        strategies.append({
            'name': 'Covered Call',
            'type': 'Income Strategy',
            'description': f'Buy 100 shares of {symbol} and sell 1 call option',
            'setup': [
                f'Buy 100 shares at ${price:.2f}',
                f'Sell 1 call option at strike ${price * 1.05:.2f} (5% OTM)',
                f'Expiration: 30-45 days'
            ],
            'capital_required': price * 100,
            'max_profit': f'${price * 100 * 0.05 + price * 0.02:.2f} (≈7%)',
            'max_loss': f'${price * 100 * 0.20:.2f} (20% decline, partially offset by premium)',
            'risk_level': 'Low-Medium',
            'ideal_market': 'Neutral to Slightly Bullish',
            'reasoning': [
                '✅ Generates consistent income from premium',
                '✅ Reduces cost basis of stock ownership',
                '✅ Limited upside but protected downside',
                f'✅ Works well in current {trend.lower()} market',
                f'⚠️ Volatility at {volatility:.1%} - moderate premium collection'
            ],
            'fit_score': 85 if trend == "Bullish" else 70
        })
        
        # Strategy 2: Cash-Secured Put (Bullish Income)
        strategies.append({
            'name': 'Cash-Secured Put',
            'type': 'Income Strategy',
            'description': f'Sell put option while holding cash to buy {symbol} if assigned',
            'setup': [
                f'Sell 1 put option at strike ${price * 0.95:.2f} (5% OTM)',
                f'Keep ${price * 100:.2f} cash reserved',
                f'Expiration: 30-45 days'
            ],
            'capital_required': price * 100,
            'max_profit': f'${price * 0.02 * 100:.2f} (2% premium, ≈24% annualized)',
            'max_loss': f'${price * 100 * 0.95:.2f} (if stock goes to zero)',
            'risk_level': 'Medium',
            'ideal_market': 'Bullish',
            'reasoning': [
                '✅ Earn premium while waiting to buy stock',
                '✅ Lower entry price if assigned',
                '✅ High probability of profit',
                f'{"✅" if trend == "Bullish" else "⚠️"} {"Perfect" if trend == "Bullish" else "Risky"} for {trend.lower()} outlook',
                f'✅ {volatility:.1%} volatility = good premium income'
            ],
            'fit_score': 90 if trend == "Bullish" else 60
        })
        
        # Strategy 3: Bull Call Spread (Moderate Bullish)
        cost = price * 0.03 * 100  # Approximate spread cost
        strategies.append({
            'name': 'Bull Call Spread',
            'type': 'Directional Strategy',
            'description': f'Buy call at lower strike, sell call at higher strike on {symbol}',
            'setup': [
                f'Buy 1 call at ${price:.2f} (ATM)',
                f'Sell 1 call at ${price * 1.10:.2f} (10% OTM)',
                f'Expiration: 30-60 days'
            ],
            'capital_required': cost,
            'max_profit': f'${(price * 0.10 - price * 0.03) * 100:.2f} (≈{((price * 0.10 - price * 0.03) / (price * 0.03)) * 100:.0f}% return)',
            'max_loss': f'${cost:.2f} (limited to net debit)',
            'risk_level': 'Medium',
            'ideal_market': 'Moderately Bullish',
            'reasoning': [
                '✅ Defined risk and reward',
                '✅ Lower cost than buying calls outright',
                '✅ Profits from moderate price increase',
                f'{"✅" if trend == "Bullish" else "⚠️"} Aligns with {trend.lower()} trend',
                f'⚠️ Volatility {volatility:.1%} - best if vol decreases'
            ],
            'fit_score': 88 if trend == "Bullish" else 55
        })
        
        # Strategy 4: Iron Condor (Neutral, High Probability)
        strategies.append({
            'name': 'Iron Condor',
            'type': 'Income Strategy',
            'description': f'Profit from {symbol} staying in a range',
            'setup': [
                f'Sell put at ${price * 0.90:.2f} (10% OTM)',
                f'Buy put at ${price * 0.85:.2f} (15% OTM)',
                f'Sell call at ${price * 1.10:.2f} (10% OTM)',
                f'Buy call at ${price * 1.15:.2f} (15% OTM)',
                f'Expiration: 30-45 days'
            ],
            'capital_required': price * 100 * 0.05,
            'max_profit': f'${price * 0.02 * 100:.2f} (premium collected)',
            'max_loss': f'${price * 100 * 0.05 - price * 0.02 * 100:.2f} (width of spread - premium)',
            'risk_level': 'Medium',
            'ideal_market': 'Neutral (Low Volatility)',
            'reasoning': [
                '✅ High probability of profit (≈70%)',
                '✅ Profits from time decay',
                '✅ Defined risk on both sides',
                f'{"✅" if volatility < 0.35 else "⚠️"} Volatility {volatility:.1%} - {"ideal" if volatility < 0.35 else "high"} for this strategy',
                '⚠️ Requires active management'
            ],
            'fit_score': 80 if volatility < 0.35 else 65
        })
        
        # Strategy 5: Long Straddle (High Volatility Play)
        straddle_cost = price * 0.05 * 100 * 2  # Both call and put
        strategies.append({
            'name': 'Long Straddle',
            'type': 'Volatility Strategy',
            'description': f'Profit from large move in either direction on {symbol}',
            'setup': [
                f'Buy 1 call at ${price:.2f} (ATM)',
                f'Buy 1 put at ${price:.2f} (ATM)',
                f'Expiration: 30-60 days'
            ],
            'capital_required': straddle_cost,
            'max_profit': 'Unlimited (if large move occurs)',
            'max_loss': f'${straddle_cost:.2f} (if price unchanged at expiration)',
            'risk_level': 'High',
            'ideal_market': 'High Volatility / Big Move Expected',
            'reasoning': [
                '✅ Profits from large moves in either direction',
                '✅ No directional bias needed',
                '⚠️ Expensive strategy (high premium cost)',
                f'{"⚠️" if volatility > 0.40 else "✅"} Volatility {volatility:.1%} - {"expensive" if volatility > 0.40 else "reasonable"} premium',
                '⚠️ Needs significant move to be profitable'
            ],
            'fit_score': 70 if volatility < 0.40 else 50
        })
        
        # Sort strategies by fit score
        strategies.sort(key=lambda x: x['fit_score'], reverse=True)
        
        # Add risk assessment using the risk manager
        for strategy in strategies:
            risk_score = self._calculate_risk_score(
                strategy['capital_required'],
                liquidity,
                max_risk_pct
            )
            strategy['risk_assessment'] = risk_score
        
        # Enhance top strategies with LLAMA AI insights
        top_strategies = strategies[:5]
        
        if self.llama_enhancer.enabled:
            print("🤖 Generating AI-powered insights...")
            
            # Get market insights (once for all strategies)
            market_insights = self.llama_enhancer.get_market_insights(stock_data, user_params)
            
            # Enhance each top strategy
            for i, strategy in enumerate(top_strategies):
                if i < 3:  # Enhance top 3 strategies to save API calls
                    ai_reasoning = self.llama_enhancer.enhance_strategy_reasoning(
                        strategy, stock_data, user_params
                    )
                    if ai_reasoning:
                        strategy['ai_reasoning'] = ai_reasoning
            
            # Get risk assessment commentary
            risk_commentary = self.llama_enhancer.get_risk_assessment_commentary(
                top_strategies, user_params
            )
            
            # Add LLAMA insights to result
            for strategy in top_strategies:
                strategy['market_insights'] = market_insights
                strategy['risk_commentary'] = risk_commentary
        
        return top_strategies  # Return top 5 with AI enhancements
    
    def _calculate_risk_score(self, capital_required, liquidity, max_risk_pct):
        """Calculate risk assessment for a strategy"""
        
        if capital_required > liquidity:
            return {
                'status': '❌ INSUFFICIENT CAPITAL',
                'message': f'Requires ${capital_required:.2f} but only ${liquidity:.2f} available'
            }
        
        position_pct = (capital_required / liquidity) * 100
        
        if position_pct > max_risk_pct:
            return {
                'status': '⚠️ EXCEEDS RISK LIMIT',
                'message': f'Uses {position_pct:.1f}% of portfolio (limit: {max_risk_pct}%)'
            }
        
        if position_pct < max_risk_pct * 0.5:
            return {
                'status': '✅ WITHIN RISK PARAMETERS',
                'message': f'Uses {position_pct:.1f}% of portfolio - Conservative sizing'
            }
        
        return {
            'status': '✅ ACCEPTABLE RISK',
            'message': f'Uses {position_pct:.1f}% of portfolio'
        }


class OptionsTitanGUI:
    """Main GUI application for OptionsTitan"""
    
    def __init__(self, root):
        self.root = root
        title_suffix = " (LLAMA AI Enhanced)" if LLAMA_AVAILABLE else ""
        self.root.title(f"OptionsTitan - AI Options Strategy Analyzer{title_suffix}")
        self.root.geometry("1200x800")
        self.root.configure(bg='#1e1e1e')
        
        # Initialize analyzer
        if DEPENDENCIES_AVAILABLE:
            self.analyzer = OptionsStrategyAnalyzer()
        else:
            self.analyzer = None
        
        # Configure style
        self.setup_style()
        
        # Create GUI components
        self.create_widgets()
        
    def setup_style(self):
        """Configure ttk styles for modern look"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure colors
        bg_color = '#1e1e1e'
        fg_color = '#ffffff'
        accent_color = '#0078d4'
        
        style.configure('Title.TLabel', 
                       font=('Helvetica', 24, 'bold'),
                       foreground=accent_color,
                       background=bg_color)
        
        style.configure('Subtitle.TLabel',
                       font=('Helvetica', 14, 'bold'),
                       foreground=fg_color,
                       background=bg_color)
        
        style.configure('TLabel',
                       font=('Helvetica', 11),
                       foreground=fg_color,
                       background=bg_color)
        
        style.configure('TEntry',
                       font=('Helvetica', 11),
                       fieldbackground='#2d2d2d',
                       foreground=fg_color)
        
        style.configure('TButton',
                       font=('Helvetica', 12, 'bold'),
                       foreground=fg_color,
                       background=accent_color)
        
        style.map('TButton',
                 background=[('active', '#005a9e')])
        
    def create_widgets(self):
        """Create all GUI widgets"""
        
        # Main container
        main_frame = ttk.Frame(self.root, style='TFrame')
        main_frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Title
        title_label = ttk.Label(main_frame, 
                               text="🚀 OptionsTitan Strategy Analyzer",
                               style='Title.TLabel')
        title_label.pack(pady=(0, 20))
        
        # Input Frame
        input_frame = ttk.LabelFrame(main_frame, text="Strategy Parameters", padding=20)
        input_frame.pack(fill='x', pady=(0, 20))
        
        # Row 1: Stock Symbol and Liquidity
        row1 = ttk.Frame(input_frame)
        row1.pack(fill='x', pady=5)
        
        ttk.Label(row1, text="Stock Symbol:").pack(side='left', padx=(0, 10))
        self.symbol_entry = ttk.Entry(row1, width=15)
        self.symbol_entry.pack(side='left', padx=(0, 30))
        self.symbol_entry.insert(0, "SPY")
        
        ttk.Label(row1, text="Portfolio Liquidity ($):").pack(side='left', padx=(0, 10))
        self.liquidity_entry = ttk.Entry(row1, width=15)
        self.liquidity_entry.pack(side='left')
        self.liquidity_entry.insert(0, "10000")
        
        # Row 2: Risk and Target Profit
        row2 = ttk.Frame(input_frame)
        row2.pack(fill='x', pady=5)
        
        ttk.Label(row2, text="Max Risk (%):").pack(side='left', padx=(0, 10))
        self.risk_entry = ttk.Entry(row2, width=15)
        self.risk_entry.pack(side='left', padx=(0, 30))
        self.risk_entry.insert(0, "5")
        
        ttk.Label(row2, text="Target Profit (%):").pack(side='left', padx=(0, 10))
        self.profit_entry = ttk.Entry(row2, width=15)
        self.profit_entry.pack(side='left')
        self.profit_entry.insert(0, "20")
        
        # Row 3: Max Loss
        row3 = ttk.Frame(input_frame)
        row3.pack(fill='x', pady=5)
        
        ttk.Label(row3, text="Max Loss (%):").pack(side='left', padx=(0, 10))
        self.loss_entry = ttk.Entry(row3, width=15)
        self.loss_entry.pack(side='left')
        self.loss_entry.insert(0, "15")
        
        # Analyze Button
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(pady=10)
        
        self.analyze_button = ttk.Button(button_frame,
                                        text="🔍 Analyze Strategies",
                                        command=self.analyze_strategies)
        self.analyze_button.pack(side='left', padx=5)
        
        self.clear_button = ttk.Button(button_frame,
                                      text="🗑️ Clear Results",
                                      command=self.clear_results)
        self.clear_button.pack(side='left', padx=5)
        
        # Progress Label
        self.progress_label = ttk.Label(main_frame, text="", foreground='#00ff00')
        self.progress_label.pack(pady=5)
        
        # Results Frame
        results_frame = ttk.LabelFrame(main_frame, text="Strategy Recommendations", padding=10)
        results_frame.pack(fill='both', expand=True)
        
        # Scrolled text for results
        self.results_text = scrolledtext.ScrolledText(results_frame,
                                                      wrap=tk.WORD,
                                                      font=('Courier', 10),
                                                      bg='#2d2d2d',
                                                      fg='#ffffff',
                                                      insertbackground='#ffffff')
        self.results_text.pack(fill='both', expand=True)
        
        # Configure text tags for formatting
        self.results_text.tag_configure('header', font=('Courier', 12, 'bold'), foreground='#00d4ff')
        self.results_text.tag_configure('subheader', font=('Courier', 11, 'bold'), foreground='#00ff88')
        self.results_text.tag_configure('success', foreground='#00ff00')
        self.results_text.tag_configure('warning', foreground='#ffaa00')
        self.results_text.tag_configure('error', foreground='#ff0000')
        
        # Initial message
        self.show_welcome_message()
        
    def show_welcome_message(self):
        """Display welcome message in results area"""
        ai_status = ""
        if LLAMA_AVAILABLE:
            if hasattr(self.analyzer, 'llama_enhancer') and self.analyzer.llama_enhancer.enabled:
                ai_status = "\n✨ AI Enhancement: ACTIVE (Meta LLAMA providing intelligent insights)"
            else:
                ai_status = "\nℹ️  AI Enhancement: Available (Set LLAMA_API_KEY to enable)"
        
        welcome = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                   Welcome to OptionsTitan Strategy Analyzer                  ║
║                     AI-Powered Options Strategy Analysis                     ║
╚══════════════════════════════════════════════════════════════════════════════╝
{ai_status}

This tool helps you find the best options strategies based on:
  • Your risk tolerance and portfolio size
  • Current market conditions
  • Stock-specific analysis
  • AI-generated market insights (when enabled)

How to use:
  1. Enter your stock symbol (e.g., SPY, AAPL, TSLA)
  2. Set your portfolio liquidity (available capital)
  3. Define your risk parameters (max risk %, target profit, max loss)
  4. Click "Analyze Strategies" to get personalized recommendations

The analyzer will provide:
  ✓ Top 5 strategies ranked by fit score
  ✓ Detailed setup instructions for each strategy
  ✓ Risk/reward analysis
  ✓ Market condition alignment
  ✓ Clear reasoning for each recommendation
  ✓ AI-powered insights and commentary (when LLAMA enabled)

🤖 LLAMA AI FEATURES (when enabled):
  • Intelligent market analysis and commentary
  • Enhanced strategy reasoning tailored to your profile
  • Personalized risk assessment
  • Natural language explanations

⚠️  DISCLAIMER: This is for educational purposes only. Options trading involves 
    substantial risk. Always consult a financial advisor before trading.

Ready to start? Enter your parameters above and click "Analyze Strategies"!
"""
        self.results_text.insert('1.0', welcome)
        self.results_text.config(state='disabled')
        
    def validate_inputs(self):
        """Validate user inputs"""
        try:
            symbol = self.symbol_entry.get().strip().upper()
            liquidity = float(self.liquidity_entry.get())
            max_risk = float(self.risk_entry.get())
            target_profit = float(self.profit_entry.get())
            max_loss = float(self.loss_entry.get())
            
            if not symbol:
                raise ValueError("Stock symbol cannot be empty")
            if liquidity <= 0:
                raise ValueError("Liquidity must be positive")
            if max_risk <= 0 or max_risk > 100:
                raise ValueError("Max risk must be between 0 and 100")
            if target_profit <= 0:
                raise ValueError("Target profit must be positive")
            if max_loss <= 0:
                raise ValueError("Max loss must be positive")
            
            return symbol, liquidity, max_risk, target_profit, max_loss
        
        except ValueError as e:
            messagebox.showerror("Invalid Input", str(e))
            return None
        
    def analyze_strategies(self):
        """Main analysis function"""
        
        if not DEPENDENCIES_AVAILABLE:
            messagebox.showerror("Dependencies Missing",
                               "Required packages not installed. Please run:\n"
                               "pip install -r requirements.txt")
            return
        
        # Validate inputs
        inputs = self.validate_inputs()
        if not inputs:
            return
        
        symbol, liquidity, max_risk, target_profit, max_loss = inputs
        
        # Disable button during analysis
        self.analyze_button.config(state='disabled')
        self.progress_label.config(text="🔄 Fetching market data and analyzing strategies...")
        
        # Run analysis in separate thread to keep GUI responsive
        thread = threading.Thread(target=self._run_analysis,
                                 args=(symbol, liquidity, max_risk, target_profit, max_loss))
        thread.daemon = True
        thread.start()
        
    def _run_analysis(self, symbol, liquidity, max_risk, target_profit, max_loss):
        """Run analysis in background thread"""
        try:
            # Fetch stock data
            stock_data = self.analyzer.get_stock_data(symbol)
            
            # Generate strategies
            strategies = self.analyzer.generate_strategies(
                stock_data, liquidity, max_risk, target_profit, max_loss
            )
            
            # Display results in main thread
            self.root.after(0, self._display_results, stock_data, strategies, liquidity, max_risk)
            
        except Exception as e:
            error_msg = str(e)
            self.root.after(0, self._show_error, error_msg)
        
        finally:
            # Re-enable button
            self.root.after(0, self._reset_button)
            
    def _reset_button(self):
        """Reset analyze button state"""
        self.analyze_button.config(state='normal')
        self.progress_label.config(text="")
        
    def _show_error(self, error_msg):
        """Show error message"""
        self.progress_label.config(text="")
        messagebox.showerror("Analysis Error", error_msg)
        
    def _display_results(self, stock_data, strategies, liquidity, max_risk):
        """Display analysis results"""
        
        self.results_text.config(state='normal')
        self.results_text.delete('1.0', tk.END)
        
        # Header
        header = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                        OPTIONS STRATEGY ANALYSIS REPORT                      ║
║                        Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}                       ║
╚══════════════════════════════════════════════════════════════════════════════╝

"""
        self.results_text.insert(tk.END, header, 'header')
        
        # Market Overview
        overview = f"""
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
Risk Profile:        {'Conservative' if max_risk < 3 else 'Moderate' if max_risk < 7 else 'Aggressive'}

"""
        self.results_text.insert(tk.END, overview)
        
        # Add LLAMA AI Market Insights if available
        if strategies and 'market_insights' in strategies[0] and strategies[0]['market_insights']:
            ai_insights = f"""
🤖 AI MARKET INSIGHTS (Powered by Meta LLAMA)
{'=' * 80}
{strategies[0]['market_insights']}

"""
            self.results_text.insert(tk.END, ai_insights, 'subheader')
        
        # Strategies
        strategies_header = f"""
🎯 TOP 5 RECOMMENDED STRATEGIES
{'=' * 80}

"""
        self.results_text.insert(tk.END, strategies_header, 'subheader')
        
        for i, strategy in enumerate(strategies, 1):
            strategy_text = self._format_strategy(i, strategy)
            self.results_text.insert(tk.END, strategy_text)
            
        # Add AI Risk Commentary if available
        if strategies and 'risk_commentary' in strategies[0] and strategies[0]['risk_commentary']:
            ai_risk = f"""
🤖 AI RISK ASSESSMENT (Powered by Meta LLAMA)
{'=' * 80}
{strategies[0]['risk_commentary']}

"""
            self.results_text.insert(tk.END, ai_risk, 'success')
        
        # Footer
        footer = """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

⚠️  IMPORTANT REMINDERS:
   • Start with paper trading to practice without real money
   • Never risk more than you can afford to lose
   • Options are complex instruments - understand them before trading
   • Market conditions can change rapidly - monitor your positions
   • These are educational recommendations, not financial advice

💡 NEXT STEPS:
   1. Review each strategy's risk/reward profile
   2. Check current options chain for exact pricing
   3. Consider your market outlook and time horizon
   4. Start with the highest-scored strategy that fits your risk tolerance
   5. Always set stop-loss orders to protect your capital

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        self.results_text.insert(tk.END, footer, 'warning')
        
        # Show LLAMA status at the end
        if LLAMA_AVAILABLE and self.analyzer.llama_enhancer.enabled:
            llama_status = "\n✨ This analysis was enhanced with Meta LLAMA AI for deeper insights.\n"
            self.results_text.insert(tk.END, llama_status, 'success')
        elif LLAMA_AVAILABLE:
            llama_status = "\nℹ️  Tip: Set LLAMA_API_KEY environment variable for AI-enhanced insights.\n"
            self.results_text.insert(tk.END, llama_status)
        
        self.results_text.config(state='disabled')
        self.progress_label.config(text="✅ Analysis complete!", foreground='#00ff00')
        
    def _format_strategy(self, rank, strategy):
        """Format a single strategy for display"""
        
        stars = '⭐' * min(5, int(strategy['fit_score'] / 20))
        
        text = f"""
┌{'─' * 78}┐
│ STRATEGY #{rank}: {strategy['name']:<63} │
│ Fit Score: {strategy['fit_score']}/100 {stars:<50} │
└{'─' * 78}┘

📋 Type: {strategy['type']}
📝 Description: {strategy['description']}

💼 SETUP:
"""
        for step in strategy['setup']:
            text += f"   • {step}\n"
        
        text += f"""
💰 CAPITAL REQUIRED: ${strategy['capital_required']:,.2f}

📈 PROFIT/LOSS POTENTIAL:
   Max Profit: {strategy['max_profit']}
   Max Loss:   {strategy['max_loss']}
   Risk Level: {strategy['risk_level']}

🎯 IDEAL MARKET: {strategy['ideal_market']}

🧠 WHY THIS STRATEGY?
"""
        for reason in strategy['reasoning']:
            text += f"   {reason}\n"
        
        # Add AI-enhanced reasoning if available
        if 'ai_reasoning' in strategy and strategy['ai_reasoning']:
            text += f"""
🤖 AI INSIGHT (Meta LLAMA):
   {strategy['ai_reasoning']}
"""
        
        # Risk assessment
        risk_status = strategy['risk_assessment']['status']
        risk_msg = strategy['risk_assessment']['message']
        
        text += f"""
⚖️  RISK ASSESSMENT: {risk_status}
   {risk_msg}

{'─' * 80}

"""
        return text
        
    def clear_results(self):
        """Clear results and show welcome message"""
        self.show_welcome_message()
        self.progress_label.config(text="")


def main():
    """Main entry point"""
    
    # Check dependencies
    if not DEPENDENCIES_AVAILABLE:
        print("⚠️  Warning: Some dependencies are not installed.")
        print("Please run: pip install -r requirements.txt")
        print("\nThe GUI will still launch, but analysis features will be disabled.")
        input("Press Enter to continue...")
    
    root = tk.Tk()
    app = OptionsTitanGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
