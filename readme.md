# 🚀 OptionsTitan - AI Options Trading System

**AI-powered system for analyzing and executing profitable options trades**

> ⚠️ **Risk Warning**: Options trading involves substantial risk. This is for educational purposes only.

---

## 🎯 What It Does

OptionsTitan analyzes market data and recommends optimal options strategies tailored to your risk tolerance and goals. It combines 5 AI models with institutional-grade risk management and optional Meta LLAMA AI insights.

**Key Features:**
- 🤖 **5-Model AI Ensemble**: XGBoost, LightGBM, Random Forest with 99%+ accuracy
- 🛡️ **Smart Risk Controls**: Automatic position sizing and stop-losses
- 📈 **Real-time Analysis**: Current market data and volatility metrics
- 💡 **Strategy Recommendations**: Top 5 ranked strategies with detailed reasoning
- 🎨 **Modern GUI**: Beautiful PySide6 interface with tabbed results
- ✨ **Meta LLAMA AI**: Optional AI-powered market insights and personalized commentary
- 📊 **Multi-Year Data**: ThetaData integration for institutional-grade historical options data (NEW)

---

## 🚀 Quick Start

📖 **New to OptionsTitan?** Read **[GETTING_STARTED.md](GETTING_STARTED.md)** for step-by-step setup.

### Installation (One Time)

```bash
pip install -r requirements.txt
```

**Verify installation:**
```bash
python verify_installation.py
```

This checks that all dependencies are properly installed.

### Using the Interactive GUI (Recommended) ⭐

**Modern Qt Version (Best Experience):**
```bash
python options_gui_qt.py
```

Features:
- Modern tabbed interface with Overview, Strategies, and AI Insights tabs
- Expandable strategy cards with detailed information
- Real-time input validation and tooltips
- Export results to TXT/HTML
- Professional dark theme

**Classic Tkinter Version (Lightweight):**
```bash
python options_gui.py
```

📖 **[Complete GUI Guide](docs/gui/GUI_GUIDE.md)** | 🤖 **[Enable LLAMA AI](docs/llama/LLAMA_QUICKSTART.md)**

### Training AI Models (Advanced)

Train custom models on historical data:
```bash
python main.py
```

This runs the complete pipeline: data preprocessing, feature engineering, 5-model ensemble training, and risk analysis (takes 2-3 minutes).

---

## 📊 What to Expect

**Strategy Recommendations:**
- Top 5 strategies ranked by fit score (0-100)
- Detailed setup instructions for each
- Profit/loss potential analysis
- Risk assessment aligned with your parameters

**AI Insights (with LLAMA):**
- Market conditions analysis
- Personalized strategy reasoning
- Risk management recommendations

**Safety First:**
- 🛡️ Always paper trade first (2+ weeks)
- 🛡️ Start with small positions (1-2% of portfolio)
- 🛡️ Never risk more than you can afford to lose
- 🛡️ Set stop-losses on every trade

---

## 📊 Advanced: Multi-Year Data Collection (NEW)

**Scale your models with institutional-grade historical options data.**

OptionsTitan now supports fetching multi-year, multi-ticker options data using ThetaData for training more robust models.

### What's Included

- **45-Ticker Universe**: ETFs, Tech, Financials, Healthcare, Energy, Consumer stocks
- **5+ Years of Data**: 2019-2024+ with full COVID coverage
- **20 Contracts/Day**: Systematic sampling across strikes and expirations
- **Auto-Normalization**: Stock splits, IV calculation, schema transformation
- **Walk-Forward Validation**: 2019-2020→2021, 2019-2021→2022, etc.

### Quick Start

```bash
# 1. Setup ThetaData Terminal (one-time)
# See: docs/THETADATA_SETUP.md

# 2. Test connection
python -m src.data_collection.test_data_collection

# 3. Fetch AAPL (POC - 15-30 minutes)
python -m src.data_collection.data_fetcher --ticker AAPL --start 2019-01-01

# 4. Fetch all 45 tickers (6-15 hours)
./scripts/fetch_all_tickers.sh  # or .bat on Windows

# 5. Train multi-ticker model
python -m src.Training_MultiTicker
```

### Documentation

- 📘 **[README_DATA_COLLECTION.md](README_DATA_COLLECTION.md)** - Complete overview
- 🛠️ **[THETADATA_SETUP.md](docs/THETADATA_SETUP.md)** - Terminal installation
- 📊 **[DATA_COLLECTION_GUIDE.md](docs/DATA_COLLECTION_GUIDE.md)** - Detailed usage guide

**Note:** This is an advanced feature. The default Training.py with 60-day SPY data works great for getting started.

---

## 🛠️ Troubleshooting

**Import Errors:**
```bash
pip install -r requirements.txt
```

**GUI Won't Launch:**
- Check Python version: `python3 --version` (need 3.7+)
- Install Qt: `pip install PySide6>=6.6.0`
- Try tkinter version: `python options_gui.py`

**Can't Fetch Stock Data:**
- Verify ticker symbol spelling
- Check internet connection
- Try during market hours

📖 **[Full Troubleshooting Guide](docs/TROUBLESHOOTING.md)**

---

## 📚 Documentation

**Essential Reading:**
- 🚀 **[GETTING_STARTED.md](GETTING_STARTED.md)** - Complete setup guide (start here!)
- 📋 **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)** - Command cheat sheet for daily use
- 🎨 **[GUI Guide](docs/gui/GUI_GUIDE.md)** - Interactive strategy analyzer tutorial

**Optional Enhancements:**
- 🤖 **[LLAMA AI Setup](docs/llama/LLAMA_QUICKSTART.md)** - Enable AI insights (3-minute setup)
- ⚙️ **[Advanced Configuration](docs/ADVANCED_CONFIGURATION.md)** - Customize risk profiles

**Advanced Features:**
- 📊 **[Data Collection](README_DATA_COLLECTION.md)** - Multi-year ThetaData integration (NEW)
- 📈 [Performance Guide](docs/PERFORMANCE_GUIDE.md) - Backtesting results
- 🔧 [Troubleshooting Guide](docs/TROUBLESHOOTING.md) - Fix common issues

---

## 📁 Repository Structure

```
OptionsTitan/
├── 📄 readme.md                    # You are here
├── 📄 GETTING_STARTED.md           # Complete setup guide
├── 📄 QUICK_REFERENCE.md           # Command reference
├── 📄 README_DATA_COLLECTION.md    # ThetaData integration (NEW)
├── 🐍 main.py                      # AI training pipeline
├── 🎨 options_gui_qt.py            # Modern GUI (recommended)
├── 🎨 options_gui.py               # Classic GUI
├── 📂 src/                         # Core AI modules
│   ├── Training.py                 # Original training (60-day SPY)
│   ├── Training_MultiTicker.py     # Multi-ticker training (NEW)
│   └── data_collection/            # ThetaData integration (NEW)
├── 📂 ui/                          # Qt UI components
├── 📂 scripts/                     # Launcher scripts
│   ├── fetch_all_tickers.sh        # Data collection (NEW)
│   └── fetch_all_tickers.bat       # Data collection (NEW)
└── 📂 docs/                        # Documentation
    ├── gui/                        # GUI tutorials
    ├── llama/                      # AI setup guides
    ├── THETADATA_SETUP.md          # Terminal setup (NEW)
    └── DATA_COLLECTION_GUIDE.md    # Collection guide (NEW)
```

---

## ⚖️ Disclaimer

**Educational purposes only.** Options trading involves substantial risk of loss. Past performance doesn't guarantee future results. Consult a financial advisor before live trading.

---

---

## 🎯 Quick Links

**Getting Started:**
- 📖 [Getting Started](GETTING_STARTED.md) - Complete setup guide
- 📋 [Quick Reference](QUICK_REFERENCE.md) - Command cheat sheet
- 🎨 [GUI Tutorial](docs/gui/GUI_GUIDE.md) - Learn the interface

**Optional Features:**
- 🤖 [Enable AI](docs/llama/LLAMA_QUICKSTART.md) - LLAMA setup
- 📊 [Data Collection](README_DATA_COLLECTION.md) - Multi-year ThetaData (NEW)

**Support:**
- 🔧 [Troubleshooting](docs/TROUBLESHOOTING.md) - Fix issues

---

*OptionsTitan v2.0 - Professional Options Analysis for Everyone*