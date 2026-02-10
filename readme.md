# 🚀 OptionsTitan - AI Options Trading System

**AI-powered system for analyzing and executing profitable options trades**

> ⚠️ **Risk Warning**: Options trading involves substantial risk. This is for educational purposes only.

> 🚀 **NEW**: Now with UV support! 10-100x faster installations. See [Migration Guide](docs/MIGRATION_TO_UV.md)

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
- 📊 **Free Historical Data**: 104 tickers (2008-2025) from open-source dataset with pre-calculated Greeks (NEW)

---

## 🚀 Quick Start

📖 **New to OptionsTitan?** Read **[GETTING_STARTED.md](GETTING_STARTED.md)** for step-by-step setup.

### Installation (One Time)

**With uv (recommended - 10-100x faster):**
```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Verify
uv run python verify_installation.py
```

**With pip (legacy):**
```bash
pip install -r requirements.txt
python verify_installation.py
```

This checks that all dependencies are properly installed.

### Using the Interactive GUI (Recommended) ⭐

**Modern Qt Version (Best Experience):**
```bash
# With uv
uv run python options_gui_qt.py

# Or directly
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

**Scale your models with FREE, institutional-grade historical options data.**

OptionsTitan uses the philippdubach/options-data dataset (104 tickers, 2008-2025) for training more robust models with zero cost.

### What's Included

- **104-Ticker Universe**: ETFs, indices, mega-cap tech, financials, healthcare, energy, consumer
- **17+ Years of Data**: 2008-2025 with full market cycle coverage
- **Pre-calculated Greeks**: Delta, gamma, theta, vega, rho - no computation needed
- **Remote Querying**: DuckDB/Polars with predicate pushdown - query without downloading
- **Auto-Normalization**: Stock splits handled, schema validation, quality filters
- **Walk-Forward Validation**: Multi-era training and validation support

### Quick Start

```bash
# 1. Validate free data infrastructure (6 tests)
python test_free_data_migration.py

# 2. Query sample data (SPY, Q4 2024)
python -c "
from src.data_collection.remote_query import query_remote_parquet
df = query_remote_parquet('spy', {'date_min': '2024-10-01', 'date_max': '2024-12-31'})
print(f'Retrieved {len(df):,} contracts')
"

# 3. Train multi-ticker model (requires collected data in data/processed_csv/)
python src/Training_MultiTicker.py
```

### Documentation

- 📘 **[README_DATA_COLLECTION.md](README_DATA_COLLECTION.md)** - Complete overview & migration guide
- 📊 **[DATA_COLLECTION_GUIDE.md](docs/DATA_COLLECTION_GUIDE.md)** - Detailed collection guide with examples

**Note:** This is an advanced feature. The default Training.py with 60-day SPY data works great for getting started.

---

## 🛠️ Troubleshooting

**Import Errors:**
```bash
# With uv (recommended)
uv sync

# With pip (legacy)
pip install -r requirements.txt
```

**GUI Won't Launch:**
- Check Python version: `python3 --version` (need 3.7+)
- Install Qt: `uv pip install PySide6>=6.6.0` or `pip install PySide6>=6.6.0`
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
- 📊 **[Data Collection](README_DATA_COLLECTION.md)** - FREE multi-year data (104 tickers, 2008-2025) (NEW)
- 📈 [Performance Guide](docs/PERFORMANCE_GUIDE.md) - Backtesting results
- 🔧 [Troubleshooting Guide](docs/TROUBLESHOOTING.md) - Fix common issues

---

## 📁 Repository Structure

```
OptionsTitan/
├── 📄 readme.md                    # You are here
├── 📄 GETTING_STARTED.md           # Complete setup guide
├── 📄 QUICK_REFERENCE.md           # Command reference
├── 📄 README_DATA_COLLECTION.md    # FREE data collection (NEW)
├── 🐍 main.py                      # AI training pipeline
├── 🎨 options_gui_qt.py            # Modern GUI (recommended)
├── 🎨 options_gui.py               # Classic GUI
├── 📂 src/                         # Core AI modules
│   ├── Training.py                 # Original training (60-day SPY)
│   ├── Training_MultiTicker.py     # Multi-ticker training (NEW)
│   └── data_collection/            # FREE data infrastructure (NEW)
├── 📂 ui/                          # Qt UI components
├── 📂 scripts/                     # Launcher scripts
│   ├── install.sh                  # uv installation
│   └── launch_gui.sh               # GUI launcher
├── 📂 test_free_data_migration.py  # Data validation suite
└── 📂 docs/                        # Documentation
    ├── gui/                        # GUI tutorials
    ├── llama/                      # AI setup guides
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
- 📊 [Data Collection](README_DATA_COLLECTION.md) - FREE multi-year data (NEW)

**Support:**
- 🔧 [Troubleshooting](docs/TROUBLESHOOTING.md) - Fix issues

---

*OptionsTitan v2.0 - Professional Options Analysis for Everyone*