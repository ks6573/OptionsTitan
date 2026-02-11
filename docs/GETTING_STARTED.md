# 🚀 Getting Started with OptionsTitan

Welcome! This guide will get you up and running in 5 minutes.

---

## Prerequisites

- **Python 3.7+** (check: `python3 --version`)
- **Internet connection** (for fetching market data)
- **$5,000-$10,000+** recommended portfolio size for options trading

---

## Installation (2 minutes)

### Step 1: Clone or Download

If you haven't already, get the repository:
```bash
git clone <repository-url>
cd OptionsTitan
```

Or download and extract the ZIP file.

### Step 2: Install Dependencies

**Method 1: uv (Recommended - 10-100x faster)**

```bash
# Install uv (one-time)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install OptionsTitan dependencies
uv sync

# Verify installation
uv run python verify_installation.py
```

**Method 2: pip (Legacy)**

```bash
pip install -r requirements.txt
python verify_installation.py
```

This installs:
- pandas, numpy (data processing)
- yfinance (market data)
- scikit-learn, xgboost, lightgbm (AI models)
- PySide6 (modern GUI)
- And more...

**Time:** 
- uv: ~30 seconds first time, <5 seconds after
- pip: ~2-3 minutes

---

## Choose Your Path (Pick One)

### Path A: Interactive GUI (Recommended for Beginners) ⭐

**Best for:** Visual interface, exploring strategies, learning

**Launch the modern Qt GUI:**
```bash
# Mac/Linux
./scripts/launch_gui_qt.sh

# Windows
scripts\launch_gui_qt.bat

# Or directly (with uv)
uv run python options_gui_qt.py

# Or without uv
python options_gui_qt.py
```

**What you'll do:**
1. Enter stock symbol (e.g., SPY, AAPL)
2. Set your portfolio size and risk tolerance
3. Click "Analyze Strategies"
4. Review 5 personalized recommendations

**[GUI Guide](docs/gui/GUI_GUIDE.md)**

---

### Path B: AI Training Pipeline (Advanced)

**Best for:** Training custom models, understanding the AI

**Run the complete training system:**
```bash
python main.py
```

**What it does:**
1. Downloads historical options data
2. Engineers 50+ features
3. Trains 5 AI models (XGBoost, LightGBM, Random Forest, etc.)
4. Generates risk metrics
5. Creates production-ready model artifacts

**Time:** 2-3 minutes

---

## Optional: Enable AI Insights 🤖

Get Meta LLAMA AI-powered insights in the GUI.

### Quick Setup:

1. **Get API Key:** Visit https://api.llama.com/
2. **Install client:** `uv pip install llama-api-client` or `pip install llama-api-client`
3. **Set key:**
   ```bash
   # Mac/Linux
   export LLAMA_API_KEY="your_key_here"
   
   # Windows
   set LLAMA_API_KEY=your_key_here
   ```

4. **Or use .env file:**
   ```bash
   cp .env.example .env
   # Edit .env and add your key
   ```

📖 **[LLAMA Setup Guide](docs/llama/LLAMA_QUICKSTART.md)**

---

## Verify Installation

### Quick Verification (Recommended):
```bash
# With uv
uv run python verify_installation.py

# Or directly
python verify_installation.py
```

This checks all dependencies and confirms everything is installed correctly.

### Manual Test - GUI:
```bash
# With uv
uv run python options_gui_qt.py

# Or directly
python options_gui_qt.py
```

Expected: Window opens with "OptionsTitan Strategy Analyzer" title

### Manual Test - Training Pipeline:
```bash
python main.py
```

Expected: See "Starting OptionsTitan AI Options Trading System..." message

---

## Your First Analysis

### GUI Method:

1. **Launch GUI:** `python options_gui_qt.py`

2. **Enter parameters:**
   - Symbol: `SPY`
   - Liquidity: `$10,000`
   - Max Risk: `5%`
   - Target Profit: `20%`
   - Max Loss: `15%`

3. **Click:** "🔍 Analyze Strategies"

4. **Review results** in the three tabs:
   - Overview: Market summary
   - Strategies: Detailed recommendations
   - AI Insights: LLAMA commentary (if enabled)

5. **Take action:**
   - Paper trade the top-ranked strategy
   - Start with minimum position sizes
   - Track your results

---

## Quick Command Reference

```bash
# Launch modern Qt GUI (Recommended)
python options_gui_qt.py

# Launch classic Tkinter GUI
python options_gui.py

# Train AI models
python main.py

# Enable LLAMA AI
./scripts/setup_llama.sh

# View documentation
cat readme.md
cat QUICK_REFERENCE.md
```

---

## What To Read Next

### For GUI Users:
1. ✅ **You're ready!** Start analyzing strategies
2. 📖 [GUI Guide](docs/gui/GUI_GUIDE.md) - Learn all features
3. 🤖 [LLAMA Setup](docs/llama/LLAMA_QUICKSTART.md) - Enable AI

### For AI Pipeline Users:
1. 📊 [Performance Guide](docs/PERFORMANCE_GUIDE.md) - What to expect
2. ⚙️ [Advanced Config](docs/ADVANCED_CONFIGURATION.md) - Customize settings
3. 🔧 [Troubleshooting](docs/TROUBLESHOOTING.md) - Common issues

### For Troubleshooting:
1. 🔧 [Troubleshooting Guide](docs/TROUBLESHOOTING.md) - Fix common issues
2. 📋 [Quick Reference](QUICK_REFERENCE.md) - Essential commands

---

## ⚠️ Important Safety Reminders

### Before Live Trading:

1. **Paper Trade First**
   - Practice with fake money for 2+ weeks
   - Understand how strategies work
   - Build confidence in the system

2. **Start Small**
   - Use minimum position sizes
   - Risk only 1-2% per trade initially
   - Scale up as you gain experience

3. **Understand Options**
   - Options can expire worthless
   - Time decay affects value
   - Volatility impacts pricing
   - Learn the Greeks (Delta, Theta, Vega, Gamma)

4. **Risk Management**
   - Never risk more than you can afford to lose
   - Set stop-losses on every trade
   - Diversify across uncorrelated assets
   - Keep position sizes small (2% max per trade)

5. **Legal Requirements**
   - Need $25,000+ for day trading (PDT rule)
   - Options trading approval from broker
   - Understand tax implications

### When NOT to Trade:

- ❌ During earnings announcements
- ❌ Major news events (Fed meetings, etc.)
- ❌ When drift alerts are active
- ❌ Low confidence signals (<65%)
- ❌ High VIX periods (>35)
- ❌ When you're emotional or stressed

---

## 💡 Pro Tips

1. **Focus on High-Confidence Signals**
   - Only trade strategies with 75%+ fit scores
   - Higher scores = better alignment with conditions

2. **Check Market Context**
   - Review the AI market insights
   - Understand the current trend
   - Consider overall market conditions (VIX, SPY trend)

3. **Position Sizing**
   - Use the recommended position sizes
   - Never exceed your max risk percentage
   - Scale down if uncertain

4. **Track Performance**
   - Keep a trading journal
   - Record entry/exit prices
   - Note what worked and what didn't
   - Review weekly

5. **Continuous Learning**
   - Read the documentation
   - Understand each strategy
   - Learn from both wins and losses
   - Stay updated on options education

---

## 🆘 Need Help?

### Quick Fixes:

**"Import errors"**
```bash
uv sync                           # recommended
pip install -r requirements.txt   # legacy
```

**"No data for symbol"**
- Verify ticker spelling
- Try during market hours
- Check internet connection

**"GUI won't launch"**
- Verify Python 3.7+
- Check PySide6 installed: `uv pip install PySide6` or `pip install PySide6`
- Try tkinter version: `python options_gui.py`

### Detailed Help:

- 🔧 [Troubleshooting Guide](docs/TROUBLESHOOTING.md)
- 📖 [Full Documentation](docs/gui/GUI_GUIDE.md)
- 📋 [Quick Reference](QUICK_REFERENCE.md)

---

## 📚 Documentation Overview

| Document | Purpose | When to Read |
|----------|---------|--------------|
| **readme.md** | Project overview | Start here |
| **GETTING_STARTED.md** | Setup guide | You are here |
| **QUICK_REFERENCE.md** | Command cheat sheet | Daily use |
| **docs/gui/GUI_GUIDE.md** | Complete GUI guide | Using GUI |
| **docs/llama/LLAMA_AI_SETUP.md** | AI enhancement | Enable AI |
| **docs/PERFORMANCE_GUIDE.md** | Expected results | Strategy planning |
| **docs/TROUBLESHOOTING.md** | Fix issues | When stuck |

---

## ✅ You're Ready!

You now have:
- ✅ All dependencies installed
- ✅ GUI ready to use
- ✅ Training pipeline configured
- ✅ Documentation available

**Next step:** Launch the GUI and run your first analysis!

```bash
uv run python options_gui_qt.py  # or just: python options_gui_qt.py
```

**Remember:** This is for educational purposes. Paper trade first, start small, and always manage your risk!

---

## 🎯 Quick Start Paths Summary

### Absolute Beginner
1. Install dependencies
2. Launch GUI: `python options_gui_qt.py`
3. Read GUI guide
4. Paper trade for 2 weeks

### Experienced Trader
1. Install dependencies
2. Launch GUI: `python options_gui_qt.py`
3. Run analysis
4. Implement top strategy

### Data Scientist / Developer
1. Install dependencies
2. Train models: `python main.py`
3. Review architecture docs
4. Customize settings

---

*Happy trading! 🚀*

*OptionsTitan v2.0 - Professional Options Analysis Made Simple*
