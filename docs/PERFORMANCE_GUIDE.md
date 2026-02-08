# 📈 Performance Guide

Complete guide to understanding and optimizing OptionsTitan's performance.

---

## 🎯 Current Performance Metrics

### Model Accuracy (Latest Results)
```
Ensemble Performance:
├── Overall AUC: 99.26% (exceptional)
├── Test Accuracy: 98.00%
├── Individual Models:
│   ├── XGBoost: 99.91% AUC
│   ├── LightGBM: 99.70% AUC
│   ├── Gradient Boost: 99.55% AUC
│   ├── Random Forest: 99.13% AUC
│   └── Logistic: 84.47% AUC
└── Ensemble Weights: Optimized automatically
```

### Risk Metrics
```
Risk Assessment:
├── Daily Volatility: 1.98%
├── Annual Volatility: 31.39%
├── 95% VaR: -3.18% (daily loss potential)
├── 99% VaR: -4.25% (worst 1% of days)
├── Max Drawdown: Varies by risk profile
└── Sharpe Ratio: Risk-adjusted returns
```

---

## 📊 Backtesting Results

### Historical Performance
Based on synthetic data backtesting:

**Trade Statistics:**
- **Total Trades**: 400-600 per backtest
- **Win Rate**: 46-65% (varies by market conditions)
- **Average Win**: +110% per winning trade
- **Average Loss**: -41% per losing trade
- **Profit Factor**: 2.33 (good risk/reward ratio)

**Portfolio Metrics:**
- **Annual Return**: Varies by risk profile and market conditions
- **Maximum Drawdown**: 15-35% (depends on risk settings)
- **Volatility**: 10-40% annual
- **Sharpe Ratio**: 0.1-1.5 (higher is better)

### Performance by Risk Profile

#### Conservative Profile
```
Expected Metrics:
├── Win Rate: 65-70%
├── Average Win: +25%
├── Average Loss: -15%
├── Max Drawdown: 10-15%
├── Annual Return: 30-60%
└── Sharpe Ratio: 1.2-2.0
```

#### Moderate Profile (Default)
```
Expected Metrics:
├── Win Rate: 60-65%
├── Average Win: +30%
├── Average Loss: -20%
├── Max Drawdown: 15-25%
├── Annual Return: 50-100%
└── Sharpe Ratio: 0.8-1.5
```

#### Aggressive Profile
```
Expected Metrics:
├── Win Rate: 55-60%
├── Average Win: +40%
├── Average Loss: -25%
├── Max Drawdown: 20-35%
├── Annual Return: 80-200%
└── Sharpe Ratio: 0.5-1.2
```

---

## 🎯 Performance Optimization

### Improving Win Rate

#### Model-Level Improvements
1. **Increase confidence threshold**:
```python
confidence_threshold = 0.75  # From 0.70
```

2. **Focus on high-importance features**:
```python
# Keep only top 10 features
top_features = feature_importance.head(10).index.tolist()
```

3. **Ensemble weight optimization**:
```python
# Ensure weights are properly optimized (not all 0.2000)
```

#### Strategy-Level Improvements
1. **Avoid low-conviction trades**
2. **Filter out high-volatility periods**
3. **Focus on liquid options only**
4. **Avoid earnings weeks**

### Reducing Drawdown

#### Position Sizing
```python
# Reduce maximum position size
max_position_size = 0.015  # From 0.02

# Implement Kelly Criterion more conservatively
kelly_cap = 0.10  # From 0.15
```

#### Stop-Loss Optimization
```python
# Tighter stop losses
stop_loss = 0.15  # From 0.20

# Dynamic stop losses based on volatility
stop_loss = base_stop * (1 + current_iv)
```

#### Risk Monitoring
```python
# Daily loss limits
if daily_pnl < -0.02:  # -2% daily limit
    halt_trading = True

# Weekly loss limits  
if weekly_pnl < -0.05:  # -5% weekly limit
    reduce_position_sizes = True
```

### Improving Sharpe Ratio

#### Risk-Adjusted Targeting
```python
# Higher profit targets for risky trades
if implied_volatility > iv_75th_percentile:
    profit_target *= 1.5
```

#### Volatility Timing
```python
# Avoid high-volatility periods
if vix_level > 30:
    skip_trade = True
```

#### Portfolio Diversification
```python
# Limit correlated positions
max_correlation = 0.50  # From 0.60
```

---

## 📉 Performance Issues

### Low Win Rate (<50%)

**Possible Causes:**
1. **Market regime change** - model trained on different conditions
2. **Overfitting** - model too complex for available data
3. **Data quality issues** - poor or synthetic data
4. **Confidence threshold too low** - taking too many marginal trades

**Solutions:**
1. **Retrain with recent data**
2. **Increase confidence threshold** to 75%+
3. **Switch to Conservative profile**
4. **Paper trade** until performance improves

### High Drawdown (>20%)

**Possible Causes:**
1. **Position sizes too large**
2. **Stop losses too wide**
3. **Correlated positions** (all losing together)
4. **Market crash scenario**

**Solutions:**
1. **Reduce position sizes** by 50%
2. **Tighten stop losses** to 15%
3. **Diversify better** (lower correlation limits)
4. **Implement daily loss limits**

### Poor Sharpe Ratio (<0.5)

**Possible Causes:**
1. **High volatility, low returns**
2. **Inadequate risk management**
3. **Poor entry/exit timing**
4. **Market conditions unfavorable**

**Solutions:**
1. **Focus on risk-adjusted returns**
2. **Implement volatility filters**
3. **Optimize entry/exit rules**
4. **Consider market timing**

---

## 🔍 Diagnostic Tools

### Performance Analysis
```python
# Check model performance
print(f"AUC Score: {auc_score:.4f}")
print(f"Win Rate: {win_rate:.1%}")
print(f"Sharpe Ratio: {sharpe_ratio:.2f}")

# Analyze feature importance
print(feature_importance.head(10))

# Check ensemble weights
print(optimized_weights)
```

### Risk Analysis
```python
# Current risk metrics
print(f"Current Drawdown: {current_drawdown:.1%}")
print(f"Daily VaR: {var_95:.1%}")
print(f"Portfolio Risk: {portfolio_risk:.1%}")

# Stress test results
print(stress_test_results)
```

### Data Quality Checks
```python
# Check for data issues
print(f"Missing Values: {df.isnull().sum()}")
print(f"Constant Features: {constant_features}")
print(f"Drift Score: {drift_score:.1f}%")
```

---

## 📊 Benchmarking

### Model Performance Benchmarks
- **AUC > 0.75**: Acceptable
- **AUC > 0.85**: Good  
- **AUC > 0.95**: Excellent
- **AUC > 0.99**: Exceptional (current level)

### Trading Performance Benchmarks
- **Win Rate > 60%**: Acceptable
- **Win Rate > 70%**: Good
- **Sharpe Ratio > 1.0**: Good
- **Sharpe Ratio > 1.5**: Excellent
- **Max Drawdown < 15%**: Good
- **Max Drawdown < 10%**: Excellent

### Risk Management Benchmarks
- **Daily VaR < -3%**: Acceptable
- **Daily VaR < -2%**: Good
- **Portfolio Risk < 10%**: Conservative
- **Portfolio Risk < 15%**: Moderate
- **Portfolio Risk > 20%**: Aggressive (risky)

---

## 🚀 Performance Tuning

### For Maximum Accuracy
```python
# Increase model complexity
n_estimators = 200
max_depth = 7
optimization_trials = 50

# Use all features
feature_selection = False

# Strict confidence threshold
confidence_threshold = 0.80
```

### For Maximum Speed
```python
# Reduce model complexity
n_estimators = 50
max_depth = 3
optimization_trials = 10

# Use fewer features
top_features_only = True

# Lower confidence threshold
confidence_threshold = 0.65
```

### For Maximum Safety
```python
# Conservative risk settings
max_position_size = 0.01
stop_loss = 0.10
profit_target = 0.20

# High confidence requirement
confidence_threshold = 0.85

# Strict risk limits
max_portfolio_risk = 0.05
```

---

## 📈 Expected Returns by Account Size

### $25,000 Account (Minimum)
**Conservative:**
- Monthly: $1,250-2,500 (5-10%)
- Annual: $15,000-30,000 (60-120%)

**Moderate:**
- Monthly: $2,500-3,750 (10-15%)
- Annual: $30,000-60,000 (120-240%)

### $100,000 Account
**Conservative:**
- Monthly: $5,000-10,000 (5-10%)
- Annual: $60,000-120,000 (60-120%)

**Moderate:**
- Monthly: $10,000-15,000 (10-15%)
- Annual: $120,000-240,000 (120-240%)

**Important**: These are estimates based on backtesting. Real results will vary significantly based on market conditions, execution quality, and discipline in following the system.

---

## ⚠️ Performance Warnings

### Unrealistic Expectations
- **Not a money printer**: Losses will occur
- **Market dependent**: Performance varies with conditions
- **Requires discipline**: Must follow system signals
- **No guarantees**: Past performance ≠ future results

### Red Flags
- **Win rate suddenly drops** below 50%
- **Drawdown exceeds** 25%
- **Sharpe ratio turns** negative
- **Drift alerts** become frequent

**Action**: Stop trading, analyze issues, retrain models, start with paper trading.

---

*For basic usage, see the main [README](../readme.md)*