# 📋 OptionsTitan Quick Reference

**One-page reference for daily use**

---

## 🚀 Essential Commands

```bash
# Install and verify
uv sync                            # recommended (fast)
pip install -r requirements.txt    # legacy (slower)
uv run python verify_installation.py  # Check all dependencies

# Run (with uv)
uv run python main.py              # Train models
uv run python options_gui_qt.py    # Modern GUI
uv run python options_gui.py       # Classic GUI

# Or run directly (if venv activated)
python main.py                     # Train models
python options_gui_qt.py           # Modern GUI
python options_gui.py              # Classic GUI

# Check results
ls -la models/          # Trained models
tail logs/*.log         # Recent logs
```

---

## 📊 Key Metrics to Monitor

| Metric | Good | Warning | Action |
|--------|------|---------|--------|
| **Confidence** | >70% | 60-70% | <60% skip trade |
| **Drift Score** | <20% | 20-30% | >30% retrain |
| **Win Rate** | >60% | 50-60% | <50% reduce size |
| **Drawdown** | <10% | 10-15% | >15% stop trading |
| **VIX Level** | <25 | 25-35 | >35 avoid trading |

---

## 🎯 Risk Profiles

| Profile | Position | Profit | Stop | Confidence |
|---------|----------|--------|------|------------|
| **Conservative** | 1.5% | 25% | 15% | 75% |
| **Moderate** | 2.0% | 30% | 20% | 70% |
| **Aggressive** | 3.0% | 40% | 25% | 65% |

*Change in `src/Training.py`: `SELECTED_RISK_PROFILE = MODERATE_CONFIG`*

---

## 🚨 Emergency Actions

| Alert | Immediate Action |
|-------|------------------|
| **Drift Detected** | `python main.py` (retrain) |
| **High Drawdown** | Reduce position sizes 50% |
| **Low Win Rate** | Switch to Conservative profile |
| **System Error** | Check `logs/` folder |

---

## 📁 File Locations

| What | Where |
|------|-------|
| **Run System** | `python main.py` |
| **Main Code** | `src/Training.py` |
| **Trained Models** | `models/*.pkl` |
| **Execution Logs** | `logs/*.log` |
| **Documentation** | `docs/*.md` |

---

## 🎯 Success Checklist

**Before Live Trading:**
- [ ] Paper traded for 2+ weeks
- [ ] Win rate >60% consistently  
- [ ] Understand all risk controls
- [ ] Have $25,000+ account
- [ ] Set up stop-loss discipline

**Daily Routine:**
- [ ] Run `uv run python main.py` or `python main.py`
- [ ] Check confidence levels
- [ ] Review risk metrics
- [ ] Execute high-confidence trades
- [ ] Monitor positions

---

*For complete guides, see [docs/](docs/) folder*