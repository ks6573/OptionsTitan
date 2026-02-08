# ⚙️ Advanced Configuration Guide

This guide covers advanced configuration options for OptionsTitan.

---

## 🎚️ Risk Profiles

Choose your risk tolerance by editing `src/Training.py`:

### Conservative Profile
```python
SELECTED_RISK_PROFILE = CONSERVATIVE_CONFIG
```
- **Position Size**: 1.5% of portfolio per trade
- **Profit Target**: 25%
- **Stop Loss**: 15%
- **Confidence Threshold**: 75% (higher confidence required)

### Moderate Profile (Default)
```python
SELECTED_RISK_PROFILE = MODERATE_CONFIG
```
- **Position Size**: 2% of portfolio per trade
- **Profit Target**: 30%
- **Stop Loss**: 20%
- **Confidence Threshold**: 70%

### Aggressive Profile
```python
SELECTED_RISK_PROFILE = AGGRESSIVE_CONFIG
```
- **Position Size**: 3% of portfolio per trade
- **Profit Target**: 40%
- **Stop Loss**: 25%
- **Confidence Threshold**: 65%

---

## 🔧 Risk Settings

Edit these parameters in `src/Training.py`:

### Position Sizing
```python
max_position_size = 0.02      # 2% of portfolio per trade
max_portfolio_risk = 0.10     # 10% maximum portfolio risk
max_correlation = 0.60        # 60% maximum correlation between positions
```

### Profit/Loss Targets
```python
profit_target = 0.30          # Take profits at 30%
stop_loss = 0.20              # Stop losses at 20%
```

### Model Confidence
```python
confidence_threshold = 0.70   # Only trade above 70% confidence
```

---

## 📊 Advanced Settings

### Drift Detection
```python
drift_threshold = 0.10        # 10% p-value threshold for drift detection
quality_threshold = 0.95      # 95% minimum data quality score
```

### Feature Selection
- **Variance Threshold**: 0.01 minimum variance required
- **Automatic Removal**: Non-predictive indicators removed automatically
- **Feature Importance**: AI-weighted based on ensemble performance

### Ensemble Weights
- **Multi-objective Optimization**: 70% AUC + 20% diversity + 10% stability
- **Automatic Rebalancing**: Weights optimized for each training run
- **Model Correlation**: Prevents over-concentration in similar models

---

## 🎯 Portfolio-Level Controls

### Risk Limits
```python
position_limits = {
    'max_position_size': 0.02,        # 2% per position
    'max_portfolio_risk': 0.10,       # 10% total portfolio risk
    'max_correlation': 0.60,          # 60% max correlation
    'max_leverage': 2.0,              # 2x maximum leverage
    'max_sector_concentration': 0.30  # 30% max in single sector
}
```

### Stress Testing Scenarios
- **Market Crash**: -10%, -20%, -30% scenarios
- **Volatility Spike**: 2x, 3x volatility increases
- **Liquidity Crisis**: Increased spread costs
- **Tail Events**: 1st, 5th, 10th percentile historical events

---

## 🔄 Model Retraining

### Automatic Triggers
- **Drift Score** > 20%
- **Win Rate** < 60% over 20 trades
- **Drawdown** > 15%
- **Weekly Schedule** (recommended)

### Manual Retraining
```bash
python main.py  # Retrains all models automatically
```

---

## 📈 Performance Tuning

### For Speed
```python
# Reduce model complexity
n_estimators = 50        # Default: 100
max_depth = 3           # Default: 5
cv_folds = 3            # Default: 5
```

### For Accuracy
```python
# Increase model complexity
n_estimators = 200      # Default: 100
max_depth = 7           # Default: 5
optimization_trials = 50 # Default: 30
```

### For Memory
```python
# Reduce data size
sample_size = 500       # Default: 1000
batch_size = 32         # Default: 64
```

---

## 🎛️ Custom Configurations

### Creating Custom Risk Profile
```python
CUSTOM_CONFIG = {
    'max_position_size': 0.025,      # 2.5%
    'profit_target': 0.35,           # 35%
    'stop_loss': 0.18,               # 18%
    'confidence_threshold': 0.72,    # 72%
    'kelly_cap': 0.12,               # 12% Kelly cap
    'vol_sensitivity': 3.5           # Volatility sensitivity
}

SELECTED_RISK_PROFILE = CUSTOM_CONFIG
```

### Environment-Specific Settings
```python
# High volatility environment
if current_vix > 25:
    profit_target *= 0.8    # Reduce profit targets
    stop_loss *= 1.2        # Wider stop losses
    confidence_threshold += 0.05  # Higher confidence required

# Low volatility environment
if current_vix < 15:
    profit_target *= 1.2    # Higher profit targets
    stop_loss *= 0.9        # Tighter stop losses
    confidence_threshold -= 0.05  # Lower confidence acceptable
```

---

## 🚨 Advanced Warnings

### High-Risk Configurations
- **Position Size** > 5%: Extremely risky, not recommended
- **Stop Loss** > 50%: May lead to catastrophic losses
- **Confidence Threshold** < 60%: Too many low-quality signals

### Recommended Limits
- **Maximum Position Size**: 3% for aggressive traders
- **Maximum Stop Loss**: 30% in high volatility
- **Minimum Confidence**: 65% for any trading

---

## 🔍 Monitoring Advanced Metrics

### Key Performance Indicators
- **Sharpe Ratio**: > 1.0 (good), > 1.5 (excellent)
- **Sortino Ratio**: > 1.2 (good), > 2.0 (excellent)
- **Calmar Ratio**: > 0.5 (good), > 1.0 (excellent)
- **Maximum Drawdown**: < 15% (good), < 10% (excellent)

### Risk Metrics to Watch
- **VaR (95%)**: Daily loss potential
- **CVaR (95%)**: Expected loss beyond VaR
- **Correlation Matrix**: Position overlap
- **Diversification Score**: Portfolio spread

---

*For basic usage, see the main [README](../readme.md)*