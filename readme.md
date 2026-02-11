# OptionsTitan - AI Options Trading System

**AI-powered system for analyzing and executing profitable options trades.**

> **Risk Warning**: Options trading involves substantial risk. Educational purposes only.

---

## What It Does

OptionsTitan analyzes market data and recommends options strategies tailored to your risk tolerance. It combines 5 AI models with risk management and optional Meta LLAMA AI insights.

**Features:**
- 5-model ensemble (XGBoost, LightGBM, Random Forest)
- Position sizing and stop-losses
- Real-time market data and volatility metrics
- Top 5 ranked strategies with reasoning
- PySide6 GUI with tabbed interface
- Optional LLAMA AI insights
- Free historical data: 104 tickers (2008-2025), pre-calculated Greeks

---

## Quick Start

**New to OptionsTitan?** See [GETTING_STARTED.md](GETTING_STARTED.md).

### Installation

**With uv (recommended):**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
uv run python verify_installation.py
```

**With pip:**
```bash
pip install -r requirements.txt
python verify_installation.py
```

### GUI

**Qt version:**
```bash
uv run python options_gui_qt.py
```

**Tkinter version:**
```bash
python options_gui.py
```

[GUI Guide](docs/gui/GUI_GUIDE.md) | [LLAMA Setup](docs/llama/LLAMA_QUICKSTART.md)

### Training

```bash
python main.py
```

Runs preprocessing, feature engineering, 5-model training, and risk analysis (~2-3 minutes).

**Safety:** Paper trade first. Start small. Set stop-losses.

---

## Multi-Year Data Collection

Free philippdubach/options-data dataset (104 tickers, 2008-2025). No API keys.

```bash
# Validate infrastructure
python test_free_data_migration.py

# Train with auto-fetch (if data/processed_csv is empty)
python -m src.Training_MultiTicker --fetch-sample

# Or train with existing data
python src/Training_MultiTicker.py
```

[README_DATA_COLLECTION.md](README_DATA_COLLECTION.md) | [DATA_COLLECTION_GUIDE.md](docs/DATA_COLLECTION_GUIDE.md)

---

## Documentation

- [GETTING_STARTED.md](GETTING_STARTED.md) - Setup
- [QUICK_REFERENCE.md](QUICK_REFERENCE.md) - Commands
- [GUI Guide](docs/gui/GUI_GUIDE.md) - Interface
- [LLAMA Setup](docs/llama/LLAMA_QUICKSTART.md) - AI features
- [Data Collection](README_DATA_COLLECTION.md) - Multi-year data
- [Troubleshooting](docs/TROUBLESHOOTING.md) - Common issues

---

## Troubleshooting

**Import errors:** `uv sync` or `pip install -r requirements.txt`

**GUI won't launch:** Python 3.9+ required. Install PySide6: `uv pip install PySide6` or `pip install PySide6`. Fallback: `python options_gui.py`

**Stock data:** Verify ticker spelling and internet connection.

---

## Repository Structure

```
OptionsTitan/
├── main.py                 # Training pipeline
├── options_gui_qt.py       # Qt GUI
├── options_gui.py          # Tkinter GUI
├── test_free_data_migration.py
├── src/
│   ├── Training.py         # Single-ticker training
│   ├── Training_MultiTicker.py
│   └── data_collection/    # Free data infrastructure
├── ui/                     # Qt components
├── scripts/                # Launchers
└── docs/                   # Guides
```

---

## Disclaimer

Educational purposes only. Options trading involves substantial risk. Past performance does not guarantee future results. Consult a financial advisor before live trading.

---

*OptionsTitan - Professional Options Analysis*
