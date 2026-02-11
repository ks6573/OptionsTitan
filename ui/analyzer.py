"""
OptionsTitan Strategy Analyzer & LLAMA Enhancer
Core analysis logic for the Qt GUI
"""

import os

try:
    import yfinance as yf
    import numpy as np
    from src.risk_management import AdvancedRiskManager
    DEPENDENCIES_AVAILABLE = True
except ImportError:
    DEPENDENCIES_AVAILABLE = False

try:
    from llama_api_client import LlamaAPIClient
    LLAMA_AVAILABLE = True
except ImportError:
    LLAMA_AVAILABLE = False


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
                    print("⚠️  LLAMA_API_KEY not found")
            except Exception as e:
                print(f"⚠️  LLAMA API init failed: {e}")
                self.enabled = False
        else:
            print("ℹ️  LLAMA API client not installed (optional)")

    def get_market_insights(self, stock_data, user_params):
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
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"LLAMA API Error (market): {e}")
            return None

    def enhance_strategy_reasoning(self, strategy, stock_data, user_params):
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

In 2-3 sentences, explain why this strategy fits the current environment and what advantage it offers. Be concise."""

            response = self.client.chat.completions.create(
                model="Llama-4-Maverick-17B-128E-Instruct-FP8",
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"LLAMA API Error (strategy): {e}")
            return None

    def get_risk_assessment_commentary(self, strategies, user_params):
        if not self.enabled:
            return None
        try:
            prompt = f"""As a risk management expert, provide a brief assessment of this trader's options strategy approach.

Trader Profile:
- Portfolio Size: ${user_params['liquidity']:,.0f}
- Max Risk per Trade: {user_params['max_risk_pct']}%
- Target Profit: {user_params['target_profit']}%
- Max Loss Tolerance: {user_params['max_loss']}%

Top Strategy: {strategies[0]['name']} (Fit: {strategies[0]['fit_score']}/100)
Risk Level: {strategies[0]['risk_level']}

In 2-3 sentences: Assess if this risk profile is appropriate and give one key recommendation. Be supportive but realistic."""

            response = self.client.chat.completions.create(
                model="Llama-4-Maverick-17B-128E-Instruct-FP8",
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"LLAMA API Error (risk): {e}")
            return None


class OptionsStrategyAnalyzer:
    """Analyzes and generates options strategy recommendations"""

    def __init__(self):
        self.risk_manager = AdvancedRiskManager() if DEPENDENCIES_AVAILABLE else None
        self.llama_enhancer = LLAMAEnhancer()

    def get_stock_data(self, symbol, days=60):
        """Fetch stock data from Yahoo Finance."""
        from datetime import datetime, timedelta

        if not DEPENDENCIES_AVAILABLE:
            raise Exception("Dependencies not installed")

        try:
            ticker = yf.Ticker(symbol)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            hist = ticker.history(start=start_date, end=end_date)

            if hist.empty:
                raise Exception(f"No data found for symbol {symbol}")

            current_price = hist['Close'].iloc[-1]
            returns = hist['Close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252) if len(returns) > 0 else 0.30
            sma_20 = hist['Close'].rolling(window=20).mean()
            trend = "Bullish" if hist['Close'].iloc[-1] > sma_20.iloc[-1] else "Bearish"

            return {
                'symbol': symbol,
                'current_price': current_price,
                'volatility': volatility,
                'trend': trend,
                'returns': returns,
                'high_52w': hist['High'].max(),
                'low_52w': hist['Low'].min(),
                'price_history': hist[['Close']].copy(),
            }
        except Exception as e:
            raise Exception(f"Error fetching data for {symbol}: {str(e)}")

    def generate_strategies(self, stock_data, liquidity, max_risk_pct, target_profit, max_loss):
        """Generate top 5 options strategies based on user inputs."""
        symbol = stock_data['symbol']
        price = stock_data['current_price']
        volatility = stock_data['volatility']
        trend = stock_data['trend']
        user_params = {
            'liquidity': liquidity,
            'max_risk_pct': max_risk_pct,
            'target_profit': target_profit,
            'max_loss': max_loss
        }

        strategies = []

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
            'max_loss': f'${price * 100 * 0.20:.2f} (20% decline)',
            'risk_level': 'Low-Medium',
            'ideal_market': 'Neutral to Slightly Bullish',
            'reasoning': [
                '✅ Generates consistent income from premium',
                '✅ Reduces cost basis of stock ownership',
                f'✅ Works well in current {trend.lower()} market',
                f'⚠️ Volatility at {volatility:.1%} - moderate premium collection'
            ],
            'fit_score': 85 if trend == "Bullish" else 70
        })

        strategies.append({
            'name': 'Cash-Secured Put',
            'type': 'Income Strategy',
            'description': f'Sell put option while holding cash to buy {symbol} if assigned',
            'setup': [
                f'Sell 1 put at strike ${price * 0.95:.2f} (5% OTM)',
                f'Keep ${price * 100:.2f} cash reserved',
                f'Expiration: 30-45 days'
            ],
            'capital_required': price * 100,
            'max_profit': f'${price * 0.02 * 100:.2f} (2% premium)',
            'max_loss': f'${price * 100 * 0.95:.2f}',
            'risk_level': 'Medium',
            'ideal_market': 'Bullish',
            'reasoning': [
                '✅ Earn premium while waiting to buy',
                '✅ Lower entry price if assigned',
                f'{"✅" if trend == "Bullish" else "⚠️"} {"Perfect" if trend == "Bullish" else "Risky"} for {trend.lower()}',
                f'✅ {volatility:.1%} volatility = good premium'
            ],
            'fit_score': 90 if trend == "Bullish" else 60
        })

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
            'max_profit': f'${(price * 0.10 - price * 0.03) * 100:.2f}',
            'max_loss': f'${cost:.2f}',
            'risk_level': 'Medium',
            'ideal_market': 'Moderately Bullish',
            'reasoning': [
                '✅ Defined risk and reward',
                '✅ Lower cost than buying calls outright',
                f'{"✅" if trend == "Bullish" else "⚠️"} Aligns with {trend.lower()}',
                f'⚠️ Volatility {volatility:.1%} - best if vol decreases'
            ],
            'fit_score': 88 if trend == "Bullish" else 55
        })

        strategies.append({
            'name': 'Iron Condor',
            'type': 'Income Strategy',
            'description': f'Profit from {symbol} staying in a range',
            'setup': [
                f'Sell put at ${price * 0.90:.2f}, buy put at ${price * 0.85:.2f}',
                f'Sell call at ${price * 1.10:.2f}, buy call at ${price * 1.15:.2f}',
                f'Expiration: 30-45 days'
            ],
            'capital_required': price * 100 * 0.05,
            'max_profit': f'${price * 0.02 * 100:.2f}',
            'max_loss': f'${price * 100 * 0.05 - price * 0.02 * 100:.2f}',
            'risk_level': 'Medium',
            'ideal_market': 'Neutral (Low Volatility)',
            'reasoning': [
                '✅ High probability of profit (~70%)',
                '✅ Profits from time decay',
                f'{"✅" if volatility < 0.35 else "⚠️"} Volatility {volatility:.1%}',
                '⚠️ Requires active management'
            ],
            'fit_score': 80 if volatility < 0.35 else 65
        })

        straddle_cost = price * 0.05 * 100 * 2
        strategies.append({
            'name': 'Long Straddle',
            'type': 'Volatility Strategy',
            'description': f'Profit from large move in either direction on {symbol}',
            'setup': [
                f'Buy 1 call and 1 put at ${price:.2f} (ATM)',
                f'Expiration: 30-60 days'
            ],
            'capital_required': straddle_cost,
            'max_profit': 'Unlimited (if large move)',
            'max_loss': f'${straddle_cost:.2f}',
            'risk_level': 'High',
            'ideal_market': 'High Volatility Expected',
            'reasoning': [
                '✅ Profits from large moves either way',
                '✅ No directional bias needed',
                f'{"⚠️" if volatility > 0.40 else "✅"} Volatility {volatility:.1%}',
                '⚠️ Needs significant move to profit'
            ],
            'fit_score': 70 if volatility < 0.40 else 50
        })

        strategies.sort(key=lambda x: x['fit_score'], reverse=True)

        for strategy in strategies:
            strategy['risk_assessment'] = self._calculate_risk_score(
                strategy['capital_required'], liquidity, max_risk_pct
            )

        top_strategies = strategies[:5]
        if self.llama_enhancer.enabled:
            market_insights = self.llama_enhancer.get_market_insights(stock_data, user_params)
            for i, strategy in enumerate(top_strategies[:3]):
                ai_reasoning = self.llama_enhancer.enhance_strategy_reasoning(
                    strategy, stock_data, user_params
                )
                if ai_reasoning:
                    strategy['ai_reasoning'] = ai_reasoning
            risk_commentary = self.llama_enhancer.get_risk_assessment_commentary(
                top_strategies, user_params
            )
            for strategy in top_strategies:
                strategy['market_insights'] = market_insights
                strategy['risk_commentary'] = risk_commentary

        return top_strategies

    def _calculate_risk_score(self, capital_required, liquidity, max_risk_pct):
        if capital_required > liquidity:
            return {'status': '❌ INSUFFICIENT CAPITAL', 'message': f'Requires ${capital_required:.2f} but only ${liquidity:.2f} available'}
        position_pct = (capital_required / liquidity) * 100
        if position_pct > max_risk_pct:
            return {'status': '⚠️ EXCEEDS RISK LIMIT', 'message': f'Uses {position_pct:.1f}% (limit: {max_risk_pct}%)'}
        if position_pct < max_risk_pct * 0.5:
            return {'status': '✅ WITHIN RISK PARAMETERS', 'message': f'Uses {position_pct:.1f}% - Conservative'}
        return {'status': '✅ ACCEPTABLE RISK', 'message': f'Uses {position_pct:.1f}% of portfolio'}
