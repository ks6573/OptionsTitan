# 🔧 Troubleshooting Guide

Complete guide to solving common OptionsTitan issues.

---

## 🚨 Installation Issues

### ModuleNotFoundError
```bash
❌ ModuleNotFoundError: No module named 'xgboost'
```

**Solution:**
```bash
uv sync  # recommended
# or: pip install -r requirements.txt  # legacy
```

**If that fails:**
```bash
uv pip install --upgrade pip
uv pip install xgboost lightgbm scikit-learn pandas numpy yfinance ta scipy optuna
# or with pip: pip install --upgrade pip && pip install xgboost lightgbm scikit-learn pandas numpy yfinance ta scipy optuna
```

### Permission Errors
```bash
❌ Permission denied: '/usr/local/lib/python3.x/site-packages'
```

**Solution:**
```bash
uv sync  # uv handles this automatically
# or: pip install --user -r requirements.txt  # legacy
```

### Python Version Issues
```bash
❌ Python 3.7 is required, you have 3.6
```

**Solution:**
- Install Python 3.8+ from [python.org](https://python.org)
- Use `python3.8` or `python3.9` instead of `python`

---

## 🏃 Runtime Issues

### Import Errors
```bash
❌ attempted relative import with no known parent package
```

**Solution:**
- Ensure `src/__init__.py` exists
- Run from project root: `python main.py`
- Don't run individual modules directly

### Memory Errors
```bash
❌ MemoryError: Unable to allocate array
```

**Solutions:**
1. **Reduce data size** in `src/Training.py`:
```python
sample_size = 500  # Reduce from 1000
```

2. **Close other applications**
3. **Use fewer models**:
```python
models_to_use = ['xgboost', 'lightgbm']  # Instead of all 5
```

### Slow Performance
```bash
⚠️ Training taking > 10 minutes
```

**Solutions:**
1. **Reduce model complexity**:
```python
n_estimators = 50     # Reduce from 100
max_depth = 3         # Reduce from 5
optimization_trials = 10  # Reduce from 30
```

2. **Use fewer features**:
```python
# Remove computationally expensive features
enhanced_features.remove('stoch_rsi')
enhanced_features.remove('bb_width')
```

---

## 📊 Data Issues

### "Drift Detected" Warning
```bash
⚠️ SIGNIFICANT DRIFT DETECTED - MODEL RETRAINING RECOMMENDED
```

**What it means:**
- Market conditions changed significantly
- Model predictions may be less reliable

**Solutions:**
1. **Retrain immediately**:
```bash
python main.py
```

2. **Reduce position sizes temporarily**:
```python
max_position_size = 0.01  # Reduce from 0.02
```

3. **Increase confidence threshold**:
```python
confidence_threshold = 0.80  # Increase from 0.70
```

### Low Confidence Signals
```bash
📊 Confidence: 45% (below threshold)
```

**Causes:**
- High market uncertainty
- Conflicting technical indicators
- Low data quality

**Solutions:**
1. **Wait for better signals** (>70% confidence)
2. **Check VIX levels** (avoid trading when VIX >35)
3. **Review market conditions** (earnings, Fed meetings, etc.)

### Constant Features Warning
```bash
⚠️ Removed 5 constant features: ['bb_width', 'macd', ...]
```

**What it means:**
- Some technical indicators aren't varying
- Usually happens with synthetic/limited data

**Solutions:**
- **Normal behavior** with demo data
- **Use real market data** for live trading
- **Features automatically removed** - no action needed

---

## 💰 Trading Issues

### High Drawdown Alert
```bash
⚠️ Current drawdown exceeds limit: -15.3%
```

**Immediate Actions:**
1. **Stop trading** until drawdown <10%
2. **Review recent trades** for patterns
3. **Reduce position sizes** by 50%
4. **Switch to Conservative profile**

**Analysis Steps:**
1. Check win rate (should be >60%)
2. Review average loss (should be <25%)
3. Look for correlation in losing trades
4. Consider market regime change

### Risk Limit Violations
```bash
⚠️ Daily VaR exceeds limit: -4.5% < -3%
```

**Solutions:**
1. **Reduce position sizes**:
```python
max_position_size = 0.015  # Reduce from 0.02
```

2. **Tighten stop losses**:
```python
stop_loss = 0.15  # Reduce from 0.20
```

3. **Increase confidence threshold**:
```python
confidence_threshold = 0.75  # Increase from 0.70
```

### Poor Win Rate
```bash
📈 Win Rate: 45% (target: >60%)
```

**Diagnostic Steps:**
1. **Check drift status** - retrain if needed
2. **Review confidence threshold** - may be too low
3. **Analyze losing trades** - look for patterns
4. **Consider market conditions** - high volatility periods

**Solutions:**
1. **Increase confidence threshold** to 75%+
2. **Avoid low-confidence trades** entirely
3. **Focus on high-conviction signals** only
4. **Consider paper trading** to rebuild confidence

---

## 🖥️ System Issues

### File Permission Errors
```bash
❌ Permission denied: 'models/enhanced_spy_options_model.pkl'
```

**Solutions:**
```bash
chmod 755 models/
chmod 644 models/*.pkl
```

### Disk Space Issues
```bash
❌ No space left on device
```

**Solutions:**
1. **Clean old artifacts**:
```bash
rm models/production_artifacts_2024*.pkl  # Keep only recent ones
```

2. **Clear logs**:
```bash
rm logs/*.log
```

3. **Check disk space**:
```bash
df -h
```

### Network Issues
```bash
❌ Failed to fetch data from yfinance
```

**Solutions:**
1. **Check internet connection**
2. **Try again later** (yfinance rate limits)
3. **Use cached data** if available

---

## 🐛 Debug Mode

### Enable Detailed Logging
```python
# In src/Training.py, add at top:
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Check Log Files
```bash
tail -f logs/optionstitan_pipeline.log
```

### Common Log Messages

**Normal Operation:**
```
INFO - Training ensemble models...
INFO - Ensemble training completed. Final AUC: 0.9926
INFO - ✅ Phase 1 completed successfully
```

**Warning Signs:**
```
WARNING - NaN values detected in training data
WARNING - High correlation detected between positions
WARNING - Model confidence below threshold
```

**Error Indicators:**
```
ERROR - Error training gradient_boost: Input contains NaN
ERROR - Pipeline failed: 'artifacts'
ERROR - Failed to save artifacts
```

---

## 🔍 Performance Diagnostics

### Model Performance Checks
```python
# Expected ranges:
auc_score > 0.75        # Good
auc_score > 0.85        # Excellent
win_rate > 0.60         # Acceptable
win_rate > 0.70         # Good
sharpe_ratio > 1.0      # Good
max_drawdown < 0.15     # Acceptable
```

### System Health Checks
1. **Memory Usage**: <80% of available RAM
2. **CPU Usage**: <90% during training
3. **Disk Space**: >1GB free
4. **Model File Size**: 1-5MB typical

---

## 🆘 Emergency Procedures

### System Completely Broken
1. **Backup current models**:
```bash
cp -r models/ models_backup/
```

2. **Fresh installation**:
```bash
rm -rf src/__pycache__/
pip uninstall -y xgboost lightgbm scikit-learn
uv sync  # recommended
# or: pip install -r requirements.txt  # legacy
```

3. **Test with minimal config**:
```python
# Temporarily reduce complexity
sample_size = 100
n_estimators = 10
```

### Lost All Models
1. **Check backup folders**:
```bash
ls -la models/
ls -la models_backup/
```

2. **Retrain from scratch**:
```bash
python main.py
```

3. **Start with paper trading** until confidence restored

---

## 📞 Getting Help

### Before Asking for Help
1. **Check this troubleshooting guide**
2. **Review log files** in `logs/`
3. **Try the suggested solutions**
4. **Note exact error messages**

### Information to Provide
- **Error message** (exact text)
- **Python version**: `python --version`
- **Operating system**: Windows/Mac/Linux
- **Steps to reproduce** the issue
- **Log file contents** (last 20 lines)

### Community Resources
- **GitHub Issues**: Report bugs and get help
- **Documentation**: Check other docs/ files
- **Stack Overflow**: Search for similar issues

---

## ✅ Prevention Tips

### Regular Maintenance
1. **Weekly retraining** recommended
2. **Monitor drift alerts** daily
3. **Review performance metrics** weekly
4. **Update dependencies** monthly

### Best Practices
1. **Always paper trade first**
2. **Start with conservative settings**
3. **Monitor system health regularly**
4. **Keep backups of working models**
5. **Document any custom changes**

---

*For basic usage, see the main [README](../readme.md)*