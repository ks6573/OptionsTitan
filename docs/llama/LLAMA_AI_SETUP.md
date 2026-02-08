# 🤖 Meta LLAMA AI Integration Guide

## Overview

OptionsTitan's GUI can be enhanced with Meta's LLAMA AI to provide:
- **Intelligent Market Analysis**: AI-powered commentary on current market conditions
- **Enhanced Strategy Reasoning**: Personalized explanations for why strategies fit your profile
- **Risk Assessment Commentary**: Expert-level risk management insights
- **Natural Language Insights**: Easy-to-understand analysis tailored to your situation

---

## 🚀 Quick Setup

### 1. Get Your LLAMA API Key

1. Visit [Meta LLAMA API Portal](https://api.llama.com/)
2. Sign up or log in to your account
3. Navigate to API Keys section
4. Generate a new API key
5. Copy your API key (starts with `sk-llama-...`)

### 2. Install LLAMA API Client

```bash
pip install llama-api-client
```

Or add to your `requirements.txt`:
```
llama-api-client>=1.0.0
```

### 3. Set Environment Variable

**Option A: Command Line (Temporary)**

Mac/Linux:
```bash
export LLAMA_API_KEY="your_api_key_here"
python options_gui.py
```

Windows:
```cmd
set LLAMA_API_KEY=your_api_key_here
python options_gui.py
```

**Option B: .env File (Recommended)**

1. Copy the example file:
```bash
cp .env.example .env
```

2. Edit `.env` and add your key:
```
LLAMA_API_KEY=sk-llama-abc123def456...
```

3. Launch GUI normally:
```bash
python options_gui.py
```

**Option C: Python dotenv (Automatic)**

Install python-dotenv:
```bash
pip install python-dotenv
```

The GUI will automatically load `.env` if dotenv is installed.

### 4. Verify Setup

Launch the GUI and check:
- Window title shows "(LLAMA AI Enhanced)"
- Welcome message shows "AI Enhancement: ACTIVE"
- Console shows "✅ LLAMA AI Enhancement: ENABLED"

---

## 🎯 What AI Enhancement Adds

### Without AI (Standard Mode)
```
📊 MARKET OVERVIEW - SPY
Current Price: $450.25
Volatility: 18.5%
Trend: Bullish

🎯 TOP 5 RECOMMENDED STRATEGIES
STRATEGY #1: Bull Call Spread
✅ Defined risk and reward
✅ Lower cost than buying calls
```

### With AI Enhancement (LLAMA Mode)
```
📊 MARKET OVERVIEW - SPY
Current Price: $450.25
Volatility: 18.5%
Trend: Bullish

🤖 AI MARKET INSIGHTS (Powered by Meta LLAMA)
SPY is currently showing strong bullish momentum with moderate 
volatility at 18.5%, creating favorable conditions for directional 
call strategies. The price is well-positioned near recent highs, 
suggesting continued upward pressure. For your $10,000 portfolio 
with 5% risk tolerance, consider strategies that capitalize on 
this momentum while maintaining strict risk controls.

🎯 TOP 5 RECOMMENDED STRATEGIES
STRATEGY #1: Bull Call Spread

🧠 WHY THIS STRATEGY?
✅ Defined risk and reward
✅ Lower cost than buying calls

🤖 AI INSIGHT (Meta LLAMA):
This spread strategy is particularly well-suited for your moderate 
risk profile. It offers asymmetric upside exposure while capping 
your downside at just 3% of your portfolio. Given SPY's current 
bullish trend and your 20% profit target, this structure provides 
an optimal risk-reward balance.

🤖 AI RISK ASSESSMENT (Powered by Meta LLAMA)
Your 5% risk allocation per trade demonstrates prudent risk 
management for options trading. This conservative approach 
allows for multiple positions while protecting your capital. 
Key recommendation: Consider scaling position sizes based on 
the strategy's fit score - allocate more to 85+ scored trades.
```

---

## 📊 AI Features Breakdown

### 1. Market Insights
**What it does:**
- Analyzes current price, volatility, and trend
- Considers your portfolio size and risk tolerance
- Provides actionable market commentary

**When it appears:**
- Once per analysis, at the top of results
- After "YOUR PARAMETERS" section

**Example:**
> "SPY is trading in a consolidation pattern with elevated volatility. 
> For your conservative risk profile, income strategies like iron condors 
> may offer better risk-adjusted returns than directional plays."

### 2. Enhanced Strategy Reasoning
**What it does:**
- Explains why strategy fits current market
- Relates strategy to YOUR specific goals
- Highlights key advantages for your profile

**When it appears:**
- For top 3 strategies only (to manage API costs)
- In each strategy detail section

**Example:**
> "The bull call spread aligns perfectly with the current bullish trend 
> while limiting your exposure to just 3% of portfolio. Its defined risk 
> structure suits your 5% risk tolerance, and the 30-60 day timeframe 
> gives your thesis time to play out."

### 3. Risk Assessment Commentary
**What it does:**
- Evaluates your overall risk approach
- Provides personalized risk management advice
- Suggests improvements to your strategy

**When it appears:**
- Once per analysis, near the bottom
- After all strategies are displayed

**Example:**
> "Your 5% max risk per trade is appropriate for options. However, 
> consider your 15% max loss tolerance - ensure stop losses are set 
> accordingly. Diversifying across uncorrelated underlyings can further 
> reduce portfolio risk."

---

## 💰 API Costs & Management

### Understanding API Usage

Each analysis makes these API calls:
1. **Market Insights**: 1 call per analysis
2. **Enhanced Reasoning**: 3 calls (top 3 strategies)
3. **Risk Assessment**: 1 call per analysis

**Total: ~5 API calls per strategy analysis**

### Cost Optimization Tips

1. **Use Strategically**
   - Run analysis when you're seriously considering trades
   - Don't repeatedly analyze same symbol/parameters

2. **The Top-3 Limit**
   - AI enhancement only applies to top 3 strategies
   - Reduces API costs by ~40%
   - Top strategies are most relevant anyway

3. **Disable When Testing**
   - Unset API key for practice runs
   - Set it only when you need insights

4. **Monitor Usage**
   - Check your LLAMA API dashboard
   - Set usage alerts if available

### Typical Costs

Assuming Meta LLAMA pricing (check current rates):
- Per analysis: ~$0.02-0.05
- 20 analyses: ~$0.40-1.00
- 100 analyses: ~$2.00-5.00

**Much cheaper than a single options commission!**

---

## 🔧 Troubleshooting

### "LLAMA API client not installed"

**Solution:**
```bash
pip install llama-api-client
```

### "LLAMA_API_KEY not found"

**Check:**
1. Environment variable is set: `echo $LLAMA_API_KEY`
2. .env file exists and contains key
3. Restart terminal after setting variable
4. No typos in variable name (case-sensitive)

### "LLAMA API initialization failed"

**Possible causes:**
- Invalid API key
- API key expired
- Network connectivity issues
- LLAMA API service down

**Solution:**
1. Verify API key on LLAMA dashboard
2. Check internet connection
3. Try regenerating API key
4. Check LLAMA API status page

### AI Insights Not Appearing

**Check:**
1. Window title shows "(LLAMA AI Enhanced)"
2. Console shows "✅ LLAMA AI Enhancement: ENABLED"
3. Welcome message shows "AI Enhancement: ACTIVE"

**If enabled but no insights:**
- API may have hit rate limit
- Check console for error messages
- Verify API key has sufficient credits

### Rate Limit Errors

**Solution:**
- Wait a few minutes between analyses
- Check LLAMA dashboard for rate limits
- Consider upgrading API tier if needed

---

## 🎨 Customizing AI Prompts

Want to customize the AI insights? Edit `options_gui.py`:

### Market Insights Prompt
```python
# Line ~130 in LLAMAEnhancer class
def get_market_insights(self, stock_data, user_params):
    prompt = f"""As an expert options trading analyst...
    # Modify this prompt to change AI behavior
    """
```

### Strategy Reasoning Prompt
```python
# Line ~170 in LLAMAEnhancer class
def enhance_strategy_reasoning(self, strategy, stock_data, user_params):
    prompt = f"""As an options trading expert...
    # Customize strategy explanations here
    """
```

### Tips for Customization:
- Keep prompts concise (faster responses, lower costs)
- Be specific about desired output format
- Test changes with different market conditions
- Consider your target audience's expertise level

---

## 📈 Best Practices

### When to Use AI Enhancement

✅ **Use AI When:**
- Analyzing new or unfamiliar stocks
- Complex market conditions (high volatility, mixed signals)
- Learning about new strategies
- Need detailed risk assessment
- Making significant trades

❌ **Skip AI When:**
- Repeatedly analyzing same stock
- Just testing the GUI
- Very familiar with the strategy
- Want quick results without commentary

### Getting the Best Insights

1. **Be Specific with Parameters**
   - Use realistic portfolio sizes
   - Set accurate risk tolerance
   - Match profit targets to your goals

2. **Consider Market Context**
   - Run during market hours for current data
   - Note upcoming events (earnings, Fed meetings)
   - Consider broader market trends

3. **Read All Sections**
   - Market Insights set the context
   - Strategy reasoning explains the "why"
   - Risk assessment ties it together

4. **Compare Multiple Stocks**
   - Analyze 2-3 related stocks
   - Compare AI insights across tickers
   - Look for consistent themes

---

## 🔐 Security & Privacy

### API Key Safety

⚠️ **NEVER:**
- Commit `.env` to version control
- Share API key in screenshots
- Hard-code API key in scripts
- Post API key in forums/chats

✅ **ALWAYS:**
- Use environment variables or .env
- Keep .env in .gitignore
- Regenerate if accidentally exposed
- Use separate keys for testing/production

### Data Privacy

What data is sent to LLAMA API:
- Stock symbol and price data
- Volatility and trend information
- Your risk parameters (%, not actual dollars)
- Strategy details being analyzed

What is NOT sent:
- Personal information
- Account details
- Historical trading data
- Other positions or portfolio details

---

## 🆘 Support

### LLAMA API Issues
- [LLAMA API Documentation](https://docs.llama.com/)
- [LLAMA Support Portal](https://support.llama.com/)
- Check API status: [status.llama.com](https://status.llama.com/)

### OptionsTitan GUI Issues
- Check existing GitHub issues
- Review troubleshooting section
- Verify basic functionality works without AI

### Still Need Help?
1. Disable LLAMA (unset API key)
2. Test if GUI works in standard mode
3. If standard mode works, issue is API-related
4. If standard mode fails, issue is with GUI setup

---

## 🚀 Advanced Usage

### Using Multiple API Keys

For teams or high-volume users:

```bash
# Development key
export LLAMA_API_KEY_DEV="sk-llama-dev..."

# Production key
export LLAMA_API_KEY_PROD="sk-llama-prod..."

# Use in scripts
LLAMA_API_KEY=$LLAMA_API_KEY_PROD python options_gui.py
```

### Logging API Calls

Enable detailed logging:

```python
# Add to options_gui.py
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Batch Analysis Script

For analyzing multiple stocks:

```python
#!/usr/bin/env python3
import os
os.environ['LLAMA_API_KEY'] = 'your_key'

symbols = ['SPY', 'QQQ', 'IWM', 'DIA']
for symbol in symbols:
    # Run analysis programmatically
    # Save results to file
    pass
```

---

## 📚 Additional Resources

- [Meta LLAMA Documentation](https://ai.meta.com/llama/)
- [Options Strategy Guide](GUI_GUIDE.md)
- [Risk Management Best Practices](docs/ADVANCED_CONFIGURATION.md)
- [API Rate Limits & Pricing](https://api.llama.com/pricing)

---

## ✨ Feature Roadmap

Potential future AI enhancements:
- Strategy comparison (A vs B analysis)
- Multi-leg strategy builder
- Real-time adjustment suggestions
- Historical performance backtesting with AI commentary
- Portfolio-level risk analysis

---

*Last Updated: 2026-01-30*

*OptionsTitan v1.0 with Meta LLAMA AI Enhancement*
