# 🚀 GUI Quick Start - 2 Minute Guide

## Choose Your GUI Version

OptionsTitan now offers two GUI versions:

1. **PySide6 (Qt) - Recommended** ⭐
   - Modern, redesigned interface
   - Better performance and responsiveness
   - Tabbed results display
   - Enhanced visual feedback

2. **Tkinter - Classic**
   - Original interface
   - Lightweight
   - Built-in with Python

## Launch in 3 Steps

### 1. Install Dependencies (First Time Only)
```bash
pip install -r requirements.txt
```

### 2. Launch GUI

**PySide6 (Qt) Version - Recommended:**

Mac/Linux:
```bash
./scripts/launch_gui_qt.sh
```

Windows:
```cmd
scripts\launch_gui_qt.bat
```

Direct:
```bash
python options_gui_qt.py
```

**Tkinter Version (Classic):**

Mac/Linux:
```bash
./scripts/launch_gui.sh
```

Windows:
```cmd
scripts\launch_gui.bat
```

Direct:
```bash
python options_gui.py
```

### 3. Enter Your Parameters
Fill in the 5 fields:
1. **Stock Symbol**: `SPY` (or any ticker)
2. **Portfolio Liquidity**: `10000` (your available capital)
3. **Max Risk %**: `5` (risk per trade)
4. **Target Profit %**: `20` (goal)
5. **Max Loss %**: `15` (stop loss)

Click **"Analyze Strategies"** and get instant recommendations! 🎯

---

## What You'll Get

✅ **Top 5 Options Strategies** ranked by fit score
✅ **Complete Setup Instructions** for each
✅ **Profit/Loss Analysis** with specific numbers
✅ **AI Reasoning** why each strategy works
✅ **Risk Assessment** based on your parameters

---

## Quick Tips

💡 **Start Conservative**: Use 2-5% max risk initially
💡 **Test Multiple Symbols**: Try SPY, QQQ, AAPL, MSFT
💡 **Compare Scores**: Focus on strategies with 80+ fit score
💡 **Read Reasoning**: Check the "Why this strategy?" section
💡 **Paper Trade First**: Practice before using real money

---

## Example Input → Output

**Input:**
- Symbol: `SPY`
- Liquidity: `$10,000`
- Max Risk: `5%`
- Target Profit: `30%`
- Max Loss: `20%`

**Output:**
```
STRATEGY #1: Bull Call Spread ⭐⭐⭐⭐⭐
Fit Score: 88/100

Setup:
  • Buy 1 call at $450 (ATM)
  • Sell 1 call at $495 (10% OTM)
  • Expiration: 30-60 days

Capital Required: $300
Max Profit: $700 (233% return)
Max Loss: $300 (limited to debit)

Why this strategy?
  ✅ Defined risk and reward
  ✅ Lower cost than buying calls
  ✅ Aligns with bullish trend
  ✅ Volatility favorable
```

---

## Common Issues

❌ **"Dependencies Missing"**
→ Run: `pip install -r requirements.txt`

❌ **"Cannot fetch data"**
→ Check ticker symbol spelling
→ Ensure internet connection
→ Try during market hours

❌ **GUI won't open**
→ Verify Python 3.7+ installed
→ Check tkinter: `python -c "import tkinter"`
→ Install if needed (see GUI_GUIDE.md)

---

## Next Steps

📖 **Full Guide**: [GUI_GUIDE.md](GUI_GUIDE.md)
📚 **Learn Options**: [Options Playbook](https://www.optionsplaybook.com/)
🔧 **Advanced Config**: [ADVANCED_CONFIGURATION.md](docs/ADVANCED_CONFIGURATION.md)

---

⚠️ **Remember**: This is for educational purposes only. Options involve substantial risk. Always paper trade first and consult a financial advisor.

---

*Ready to analyze strategies? Launch the GUI now!* 🚀
