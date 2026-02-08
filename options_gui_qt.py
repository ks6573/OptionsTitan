#!/usr/bin/env python3
"""
OptionsTitan - PySide6 GUI for Options Strategy Analysis
A modern, redesigned interface for analyzing options strategies
Enhanced with Meta LLAMA AI for intelligent insights
"""

import sys
import os
from PySide6.QtWidgets import QApplication, QMessageBox
from PySide6.QtCore import Qt

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

# Import dependencies
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

# Import UI components
from ui.main_window import OptionsTitanMainWindow


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


class OptionsStrategyAnalyzer:
    """Analyzes and generates options strategy recommendations"""
    
    def __init__(self):
        self.risk_manager = AdvancedRiskManager()
        self.llama_enhancer = LLAMAEnhancer()
    
    def get_stock_data(self, symbol, days=60):
        """Fetch stock data from Yahoo Finance"""
        from datetime import datetime, timedelta
        
        try:
            ticker = yf.Ticker(symbol)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # Get historical data
            hist = ticker.history(start=start_date, end=end_date)
            
            if hist.empty:
                raise Exception(f"No data found for symbol {symbol}")
            
            # Get current price
            current_price = hist['Close'].iloc[-1]
            
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
            'max_loss': f'${price * 100 * 0.20:.2f} (20% decline, offset by premium)',
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
        cost = price * 0.03 * 100
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
            'max_loss': f'${price * 100 * 0.05 - price * 0.02 * 100:.2f} (width minus premium)',
            'risk_level': 'Medium',
            'ideal_market': 'Neutral (Low Volatility)',
            'reasoning': [
                '✅ High probability of profit (≈70%)',
                '✅ Profits from time decay',
                '✅ Defined risk on both sides',
                f'{"✅" if volatility < 0.35 else "⚠️"} Volatility {volatility:.1%} - {"ideal" if volatility < 0.35 else "high"}',
                '⚠️ Requires active management'
            ],
            'fit_score': 80 if volatility < 0.35 else 65
        })
        
        # Strategy 5: Long Straddle (High Volatility Play)
        straddle_cost = price * 0.05 * 100 * 2
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
            'max_loss': f'${straddle_cost:.2f} (if price unchanged)',
            'risk_level': 'High',
            'ideal_market': 'High Volatility / Big Move Expected',
            'reasoning': [
                '✅ Profits from large moves in either direction',
                '✅ No directional bias needed',
                '⚠️ Expensive strategy (high premium cost)',
                f'{"⚠️" if volatility > 0.40 else "✅"} Volatility {volatility:.1%}',
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
        
        return top_strategies
    
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


def main():
    """Main entry point for the Qt GUI"""
    
    # Check dependencies
    if not DEPENDENCIES_AVAILABLE:
        print("⚠️  Warning: Some dependencies are not installed.")
        print("Please run: pip install -r requirements.txt")
        print("\nThe GUI will still launch, but analysis features will be disabled.")
    
    # Create Qt application
    app = QApplication(sys.argv)
    app.setApplicationName("OptionsTitan")
    app.setOrganizationName("OptionsTitan")
    
    # Enable high DPI scaling
    app.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    app.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    
    # Create analyzer if dependencies available
    analyzer = OptionsStrategyAnalyzer() if DEPENDENCIES_AVAILABLE else None
    llama_enabled = analyzer.llama_enhancer.enabled if analyzer else False
    
    # Create and show main window
    window = OptionsTitanMainWindow(analyzer=analyzer, llama_enabled=llama_enabled)
    window.show()
    
    # Run application
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
