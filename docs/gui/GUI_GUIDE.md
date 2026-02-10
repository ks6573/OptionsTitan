# 🎨 OptionsTitan GUI Guide

## Interactive Options Strategy Analyzer

The OptionsTitan GUI provides a user-friendly interface to analyze options strategies based on your risk tolerance, portfolio size, and market goals.

**TWO GUI VERSIONS AVAILABLE:**

1. **PySide6 (Qt) - Recommended** ⭐
   - Modern tabbed interface with Overview, Strategies, and AI Insights tabs
   - Expandable strategy cards (click to expand/collapse)
   - Better performance with QThread background processing
   - Enhanced visual feedback and real-time validation
   - Professional dark theme

2. **Tkinter - Classic**
   - Original scrolling interface
   - Built-in with Python (no extra installation)
   - Lightweight and fast

**AI Enhancement Available** 🤖
- Get intelligent market analysis and commentary
- Receive personalized strategy explanations
- Benefit from AI-powered risk assessments
- 📖 **[LLAMA AI Setup Guide](../llama/LLAMA_AI_SETUP.md)**

---

## 🚀 Quick Start

### Launch the GUI

**PySide6 (Qt) Version - Recommended:**
```bash
./scripts/launch_gui_qt.sh
```

**Tkinter Version (Classic):**
```bash
./scripts/launch_gui.sh
```

**Option 2: Direct Python execution**
```bash
python3 options_gui.py
```

**Option 3: From Windows**
```cmd
python options_gui.py
```

---

## 📋 How to Use

### 1. **Enter Your Parameters**

The GUI has five input fields:

| Field | Description | Example |
|-------|-------------|---------|
| **Stock Symbol** | The ticker you want to trade options on | `SPY`, `AAPL`, `TSLA` |
| **Portfolio Liquidity** | Your available trading capital | `10000` ($10,000) |
| **Max Risk (%)** | Maximum portfolio percentage to risk per trade | `5` (5%) |
| **Target Profit (%)** | Your desired profit target | `20` (20%) |
| **Max Loss (%)** | Maximum acceptable loss | `15` (15%) |

### 2. **Click "Analyze Strategies"**

The system will:
- ✅ Fetch real-time market data for your stock
- ✅ Calculate volatility and trend
- ✅ Analyze 5+ different options strategies
- ✅ Rank strategies by fit score
- ✅ Provide detailed reasoning for each recommendation

### 3. **Review Results**

You'll receive:
- 📊 **Market Overview**: Current price, volatility, trend
- 🎯 **Top 5 Strategies**: Ranked by fit score (0-100)
- 💰 **Setup Instructions**: Exact steps to implement each strategy
- 📈 **Profit/Loss Analysis**: Max profit, max loss, risk level
- 🧠 **AI Reasoning**: Why each strategy is recommended
- ⚖️ **Risk Assessment**: Whether strategy fits your risk parameters

---

## 🎯 Understanding the Strategies

### 1. **Covered Call** (Conservative Income)
- **Best For**: Generating steady income, neutral to slightly bullish outlook
- **Risk Level**: Low-Medium
- **Capital Required**: High (need to own 100 shares)
- **Ideal When**: You already own the stock or want to reduce cost basis

### 2. **Cash-Secured Put** (Bullish Income)
- **Best For**: Getting paid to wait to buy stock at lower price
- **Risk Level**: Medium
- **Capital Required**: Medium (cash to buy 100 shares)
- **Ideal When**: Bullish on stock, want to enter at lower price

### 3. **Bull Call Spread** (Directional)
- **Best For**: Moderate bullish outlook with defined risk
- **Risk Level**: Medium
- **Capital Required**: Low (net debit for spread)
- **Ideal When**: Expect moderate upward move, want leverage

### 4. **Iron Condor** (Neutral Income)
- **Best For**: Profiting from range-bound movement
- **Risk Level**: Medium
- **Capital Required**: Medium (margin for spreads)
- **Ideal When**: Low volatility, expect stock to trade sideways

### 5. **Long Straddle** (Volatility Play)
- **Best For**: Expecting big move but uncertain of direction
- **Risk Level**: High
- **Capital Required**: Medium-High (both call and put premiums)
- **Ideal When**: Major event expected (earnings, FDA approval, etc.)

---

## 📊 Interpreting the Fit Score

The Fit Score (0-100) indicates how well a strategy matches your inputs and market conditions:

| Score | Interpretation | Action |
|-------|---------------|--------|
| **85-100** | Excellent fit | Strong recommendation |
| **70-84** | Good fit | Solid option |
| **60-69** | Moderate fit | Consider with caution |
| **<60** | Poor fit | Likely not suitable |

### Factors Affecting Fit Score:
- ✅ Alignment with current market trend
- ✅ Volatility levels (impacts premium pricing)
- ✅ Your risk tolerance
- ✅ Capital efficiency
- ✅ Probability of profit

---

## 🎨 GUI Features

### Main Interface

```
┌─────────────────────────────────────────────────────────┐
│  🚀 OptionsTitan Strategy Analyzer                     │
├─────────────────────────────────────────────────────────┤
│  Strategy Parameters                                    │
│  ┌───────────────────────────────────────────────────┐ │
│  │ Stock Symbol: [SPY    ]  Liquidity: [$10000   ]  │ │
│  │ Max Risk %:   [5      ]  Target %:  [20       ]  │ │
│  │ Max Loss %:   [15     ]                          │ │
│  └───────────────────────────────────────────────────┘ │
│  [🔍 Analyze Strategies]  [🗑️ Clear Results]          │
├─────────────────────────────────────────────────────────┤
│  Strategy Recommendations                               │
│  ┌───────────────────────────────────────────────────┐ │
│  │ ╔═══════════════════════════════════════════════╗ │ │
│  │ ║  STRATEGY #1: Bull Call Spread         ⭐⭐⭐⭐⭐ ║ │ │
│  │ ║  Fit Score: 88/100                           ║ │ │
│  │ ╚═══════════════════════════════════════════════╝ │ │
│  │                                                   │ │
│  │ 📋 Type: Directional Strategy                    │ │
│  │ 💰 Capital Required: $300                        │ │
│  │ 📈 Max Profit: $700 (233% return)                │ │
│  │ ...                                              │ │
│  └───────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

### Color Coding

The GUI uses intuitive color coding:
- 🟢 **Green**: Positive indicators, acceptable risk, AI insights
- 🟡 **Yellow/Orange**: Warnings, caution advised
- 🔴 **Red**: Violations, high risk
- 🔵 **Blue**: Headers and titles
- ⚪ **White**: General information

### AI Enhancement Indicators

When LLAMA AI is enabled, you'll see:
- 🤖 **AI MARKET INSIGHTS**: Intelligent market commentary
- 🤖 **AI INSIGHT**: Enhanced strategy reasoning
- 🤖 **AI RISK ASSESSMENT**: Personalized risk analysis
- ✨ **Status Message**: Confirmation of AI enhancement at bottom

---

## 💡 Tips for Best Results

### 1. **Start Conservative**
- Begin with 2-3% max risk until you're comfortable
- Use larger portfolios ($10,000+) for better flexibility
- Test with paper trading first

### 2. **Match Strategy to Market**
- **Bullish trend** → Bull Call Spread, Cash-Secured Put
- **Bearish trend** → Bear Put Spread (not in current version)
- **Neutral/Range-bound** → Iron Condor, Covered Call
- **High volatility expected** → Long Straddle

### 3. **Consider Time Frames**
- **Short-term (0-30 days)**: Higher risk, higher potential reward
- **Medium-term (30-60 days)**: Balanced approach
- **Long-term (60+ days)**: More time for thesis to play out

### 4. **Monitor Your Positions**
- Set calendar reminders to check positions
- Be ready to adjust if market conditions change
- Take profits at your target (don't be greedy)
- Cut losses at your stop (don't hope for recovery)

---

## ⚠️ Risk Warnings

### Before Trading Options:

1. **Understand the Greeks**
   - Delta: Price sensitivity
   - Theta: Time decay
   - Vega: Volatility sensitivity
   - Gamma: Delta change rate

2. **Be Aware of Risks**
   - Options can expire worthless
   - Time decay works against buyers
   - Volatility can spike unexpectedly
   - Assignment risk with short options

3. **Capital Requirements**
   - Some strategies need margin accounts
   - Spreads require approval levels
   - Know your broker's requirements

4. **Tax Implications**
   - Options profits may be short-term gains
   - Consult a tax professional
   - Keep detailed records

---

## 🔧 Troubleshooting

### "Dependencies Missing" Error

**Solution:**
```bash
uv sync  # recommended
# or: pip install -r requirements.txt  # legacy
```

### "Cannot fetch data for symbol" Error

**Causes:**
- Invalid ticker symbol
- Market closed (try during market hours)
- Network connectivity issues
- Yahoo Finance API temporarily down

**Solution:**
- Verify ticker symbol is correct
- Check internet connection
- Try again in a few minutes

### GUI Won't Launch

**Check Python Version:**
```bash
python3 --version  # Should be 3.7 or higher
```

**Check tkinter:**
```bash
python3 -c "import tkinter"
```

If tkinter missing:
- **Ubuntu/Debian**: `sudo apt-get install python3-tk`
- **macOS**: Usually included with Python
- **Windows**: Reinstall Python with tkinter option checked

### Strategies Don't Make Sense

**Remember:**
- Strategies are based on historical data and current conditions
- AI recommendations are educational, not financial advice
- Your specific situation may require different approaches
- Always do your own research

---

## 📚 Additional Resources

### Learn More About Options

- [CBOE Options Institute](https://www.cboe.com/education/)
- [Options Playbook](https://www.optionsplaybook.com/)
- [Investopedia Options Guide](https://www.investopedia.com/options-basics-tutorial-4583012)

### OptionsTitan Documentation

- [Main README](readme.md) - System overview
- [LLAMA AI Setup](../llama/LLAMA_AI_SETUP.md) - Enable AI enhancements
- [Advanced Configuration](../ADVANCED_CONFIGURATION.md) - Customize settings
- [Performance Guide](../PERFORMANCE_GUIDE.md) - Expected results
- [Troubleshooting](../TROUBLESHOOTING.md) - Common issues

---

## 🤝 Support

### Found a Bug?
- Check existing issues
- Provide detailed description
- Include error messages and screenshots

### Want a Feature?
- Describe the use case
- Explain how it would help
- Consider contributing!

---

## 📄 License & Disclaimer

**IMPORTANT**: This software is for educational purposes only. Options trading involves substantial risk of loss. Past performance does not guarantee future results.

- Not financial advice
- Use at your own risk
- Consult professionals before trading
- Test thoroughly before using real money

---

*Happy Trading! 🚀*

*OptionsTitan v1.0 - Making Options Analysis Accessible*
