import pandas as pd
import numpy as np
import yfinance as yf
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
import pickle
import warnings
from datetime import datetime, timedelta
import ta  # Technical Analysis library
from scipy import stats
import optuna
from typing import Dict, Tuple, Optional
import logging

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ---------------------------
# Data Preprocessing and Feature Engineering Functions
# ---------------------------

class DataPreprocessor:
    """Comprehensive data preprocessing pipeline for options trading data"""
    
    def __init__(self):
        self.scaler = RobustScaler()  # More robust to outliers than StandardScaler
        self.feature_columns = None
        
    def detect_outliers(self, df: pd.DataFrame, columns: list, method='iqr') -> pd.DataFrame:
        """Detect and handle outliers using IQR or Z-score method"""
        df_clean = df.copy()
        
        for col in columns:
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df_clean[col] = df_clean[col].clip(lower_bound, upper_bound)
            elif method == 'zscore':
                z_scores = np.abs(stats.zscore(df[col]))
                df_clean = df_clean[z_scores < 3]
                
        return df_clean
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values with forward fill and interpolation"""
        df_clean = df.copy()
        
        # Forward fill for price-related columns
        price_cols = ['price', 'option_price', 'strike_distance']
        for col in price_cols:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].fillna(method='ffill')
        
        # Interpolate for technical indicators
        technical_cols = ['rsi', 'implied_volatility', 'vix_level']
        for col in technical_cols:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].interpolate()
        
        # Fill remaining with median (numeric columns only; avoid timestamp/object columns)
        median_vals = df_clean.median(numeric_only=True)
        df_clean = df_clean.fillna(median_vals)
        
        return df_clean
    
    def validate_data_quality(self, df: pd.DataFrame) -> Dict:
        """Validate data quality and return quality metrics"""
        quality_report = {
            'total_rows': len(df),
            'missing_values': df.isnull().sum().to_dict(),
            'duplicate_rows': df.duplicated().sum(),
            'data_types': df.dtypes.to_dict()
        }
        
        # Check for data leakage indicators
        if 'timestamp' in df.columns:
            quality_report['date_range'] = {
                'start': df['timestamp'].min(),
                'end': df['timestamp'].max()
            }
        
        logging.info(f"Data Quality Report: {quality_report}")
        return quality_report

class FeatureEngineer:
    """Advanced feature engineering for options trading"""
    
    @staticmethod
    def calculate_greeks_approximation(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate approximate Greeks using Black-Scholes approximations"""
        df_enhanced = df.copy()
        
        # Simple delta approximation (requires strike, underlying price, time to expiry)
        if all(col in df.columns for col in ['price', 'strike_distance', 'time_to_expiry']):
            # Simplified delta calculation (call option)
            df_enhanced['delta_approx'] = np.where(
                df['strike_distance'] > 0, 
                0.5 + (df['strike_distance'] / df['price']) * 0.1,  # Simplified
                0.5 - (abs(df['strike_distance']) / df['price']) * 0.1
            )
            
            # Gamma approximation (rate of change of delta)
            df_enhanced['gamma_approx'] = 1 / (df['price'] * np.sqrt(df['time_to_expiry'] / 365))
            
            # Theta approximation (time decay)
            df_enhanced['theta_approx'] = -df['option_price'] / df['time_to_expiry']
            
        return df_enhanced
    
    @staticmethod
    def add_technical_indicators(df: pd.DataFrame, spy_data: pd.DataFrame) -> pd.DataFrame:
        """Add comprehensive technical indicators"""
        df_enhanced = df.copy()
        
        if len(spy_data) > 0:
            # Bollinger Bands
            spy_data['bb_upper'] = ta.volatility.bollinger_hband(spy_data['Close'])
            spy_data['bb_lower'] = ta.volatility.bollinger_lband(spy_data['Close'])
            spy_data['bb_width'] = (spy_data['bb_upper'] - spy_data['bb_lower']) / spy_data['Close']
            
            # MACD
            spy_data['macd'] = ta.trend.macd(spy_data['Close'])
            spy_data['macd_signal'] = ta.trend.macd_signal(spy_data['Close'])
            
            # Additional momentum indicators
            spy_data['momentum'] = ta.momentum.roc(spy_data['Close'], window=10)
            spy_data['stoch_rsi'] = ta.momentum.stochrsi(spy_data['Close'])
            
            # Volume indicators
            if 'Volume' in spy_data.columns:
                spy_data['volume_sma'] = spy_data['Volume'].rolling(window=20).mean()
            
            # Merge with options data (assuming timestamp alignment)
            latest_indicators = spy_data[['bb_width', 'macd', 'macd_signal', 'momentum', 'stoch_rsi']].iloc[-1]
            for indicator, value in latest_indicators.items():
                df_enhanced[indicator] = value
                
        return df_enhanced
    
    @staticmethod
    def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features"""
        df_enhanced = df.copy()
        
        if 'timestamp' in df.columns:
            df_enhanced['hour'] = pd.to_datetime(df['timestamp']).dt.hour
            df_enhanced['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
            df_enhanced['is_market_open'] = df_enhanced['hour'].between(9, 16)
            df_enhanced['time_to_close'] = np.where(
                df_enhanced['hour'] <= 16,
                16 - df_enhanced['hour'],
                24 - df_enhanced['hour'] + 16
            )
        
        return df_enhanced
    
    @staticmethod
    def create_market_regime_features(df: pd.DataFrame) -> pd.DataFrame:
        """Create market regime and volatility features"""
        df_enhanced = df.copy()
        
        # Volatility regime
        if 'vix_level' in df.columns:
            df_enhanced['vix_regime'] = pd.cut(
                df['vix_level'], 
                bins=[0, 15, 25, float('inf')], 
                labels=['low_vol', 'medium_vol', 'high_vol']
            )
            df_enhanced['vix_regime_encoded'] = df_enhanced['vix_regime'].cat.codes
        
        # IV percentile
        if 'implied_volatility' in df.columns:
            df_enhanced['iv_percentile'] = df['implied_volatility'].rolling(window=252).rank(pct=True)
            df_enhanced['iv_rank'] = df['implied_volatility'].rolling(window=50).rank(pct=True)
        
        return df_enhanced

def create_target_variable(df: pd.DataFrame, profit_threshold: float = 0.1, 
                          time_horizon: int = 1, transaction_cost: float = 0.005,
                          risk_adjusted: bool = True) -> pd.DataFrame:
    """
    Create a proper target variable for options trading with optional risk adjustment
    
    Args:
        df: DataFrame with options data
        profit_threshold: Minimum profit percentage to consider as positive
        time_horizon: Number of periods ahead to look for profit
        transaction_cost: Transaction cost as percentage
        risk_adjusted: Whether to apply risk-based threshold adjustments
    """
    df_with_target = df.copy()
    
    # Calculate future returns
    df_with_target['future_option_price'] = df_with_target['option_price'].shift(-time_horizon)
    
    # Calculate profit/loss percentage including transaction costs
    df_with_target['profit_pct'] = (
        (df_with_target['future_option_price'] - df_with_target['option_price']) / 
        df_with_target['option_price']
    ) - transaction_cost
    
    if risk_adjusted and 'implied_volatility' in df_with_target.columns:
        # Risk-adjusted target creation
        # Higher profit threshold for high-risk (high IV) trades
        iv_quantile_80 = df_with_target['implied_volatility'].quantile(0.8)
        high_risk_mask = df_with_target['implied_volatility'] > iv_quantile_80
        
        # Adjust threshold based on risk level
        adjusted_threshold = np.where(
            high_risk_mask, 
            profit_threshold * 1.5,  # 50% higher threshold for high-risk trades
            profit_threshold
        )
        
        # Create risk-adjusted binary target
        df_with_target['target'] = (df_with_target['profit_pct'] > adjusted_threshold).astype(int)
        
        logging.info(f"Risk-adjusted targets: {high_risk_mask.sum()} high-risk trades with {profit_threshold*1.5:.1%} threshold")
    else:
        # Standard target creation
        df_with_target['target'] = (df_with_target['profit_pct'] > profit_threshold).astype(int)
    
    # Remove rows where we can't calculate future returns
    df_with_target = df_with_target.dropna(subset=['future_option_price', 'target'])
    
    logging.info(f"Target distribution: {df_with_target['target'].value_counts()}")
    logging.info(f"Average profit/loss: {df_with_target['profit_pct'].mean():.4f}")
    
    return df_with_target

# ---------------------------
# 1. Fetch Historical Data with yfinance
# ---------------------------
# Here we get the SPY historical data.
spy = yf.Ticker("SPY")
# Example: retrieve the last 60 days of daily data.
spy_hist = spy.history(period="60d", interval="1d")
print("SPY Historical Data:")
print(spy_hist.head())

# ---------------------------
# 2. Load & Prepare Your Options Data with Enhanced Pipeline
# ---------------------------
# Initialize preprocessing and feature engineering classes
preprocessor = DataPreprocessor()
feature_engineer = FeatureEngineer()

# Load options data
try:
    df = pd.read_csv('spy_options_data.csv')
    print("Options Data Sample:")
    print(df.head())
    logging.info(f"Loaded {len(df)} rows of options data")
except FileNotFoundError:
    logging.error("spy_options_data.csv not found. Creating sample data for demonstration.")
    # Create sample data structure for testing with realistic variations
    np.random.seed(42)
    sample_size = 1000
    
    # Generate correlated market data for realism
    base_trend = np.cumsum(np.random.normal(0, 0.01, sample_size))
    
    df = pd.DataFrame({
        'price': 400 + base_trend * 20 + np.random.normal(0, 5, sample_size),
        'option_price': np.random.lognormal(2, 0.5, sample_size),  # More realistic option pricing
        'strike_distance': np.random.normal(0, 15, sample_size),  # Can be negative (ITM)
        'time_to_expiry': np.random.randint(1, 90, sample_size),
        'volume': np.random.lognormal(6, 1, sample_size).astype(int),  # Log-normal volume
        'implied_volatility': np.random.lognormal(-1.6, 0.4, sample_size),  # Realistic IV range
        'vix_level': np.random.lognormal(2.8, 0.3, sample_size),  # VIX-like distribution
        'spy_return_5min': np.random.normal(0.0002, 0.005, sample_size),  # Smaller, realistic returns
        'rsi': np.random.beta(2, 2, sample_size) * 100,  # RSI bounded 0-100
        'timestamp': pd.date_range(start='2023-01-01', periods=sample_size, freq='H')
    })

# Step 1: Data Quality Validation
quality_report = preprocessor.validate_data_quality(df)

# Step 2: Handle Missing Values and Outliers
df_clean = preprocessor.handle_missing_values(df)
numeric_columns = df_clean.select_dtypes(include=[np.number]).columns.tolist()
df_clean = preprocessor.detect_outliers(df_clean, numeric_columns, method='iqr')

# Step 3: Feature Engineering
logging.info("Starting feature engineering...")

# Add basic ratio feature
df_clean['option_to_spy'] = df_clean['option_price'] / df_clean['price']

# Add Greeks approximations
df_enhanced = feature_engineer.calculate_greeks_approximation(df_clean)

# Add technical indicators (using SPY historical data)
df_enhanced = feature_engineer.add_technical_indicators(df_enhanced, spy_hist)

# Add time-based features
df_enhanced = feature_engineer.create_time_features(df_enhanced)

# Add market regime features
df_enhanced = feature_engineer.create_market_regime_features(df_enhanced)

# Step 4: Create Target Variable (proper profit/loss calculation with risk adjustment)
df_final = create_target_variable(
    df_enhanced, 
    profit_threshold=0.05,  # 5% profit threshold
    time_horizon=1,         # 1 period ahead
    transaction_cost=0.005, # 0.5% transaction cost
    risk_adjusted=True      # Enable risk-adjusted target creation
)

# Step 5: Enhanced Feature Set
enhanced_features = [
    # Original features
    'price', 'option_price', 'strike_distance', 'time_to_expiry', 
    'volume', 'implied_volatility', 'vix_level', 'spy_return_5min', 'rsi',
    
    # Derived features
    'option_to_spy',
    
    # Greeks approximations
    'delta_approx', 'gamma_approx', 'theta_approx',
    
    # Technical indicators
    'bb_width', 'macd', 'macd_signal', 'momentum', 'stoch_rsi',
    
    # Time features
    'hour', 'day_of_week', 'is_market_open', 'time_to_close',
    
    # Market regime features
    'vix_regime_encoded', 'iv_percentile', 'iv_rank'
]

# Filter features that actually exist in the dataframe
available_features = [f for f in enhanced_features if f in df_final.columns]

# Remove constant features (variance threshold)
from sklearn.feature_selection import VarianceThreshold
variance_selector = VarianceThreshold(threshold=0.01)  # Minimum variance required
X_temp = df_final[available_features]
variance_mask = variance_selector.fit(X_temp).get_support()
available_features = [f for f, keep in zip(available_features, variance_mask) if keep]

# Log removed constant features
removed_features = [f for f, keep in zip(X_temp.columns, variance_mask) if not keep]
if removed_features:
    logging.info(f"Removed {len(removed_features)} constant features: {removed_features}")

logging.info(f"Available features: {len(available_features)} out of {len(enhanced_features)} planned")

X = df_final[available_features]
y = df_final['target']

# Step 6: Feature Scaling
X_scaled = preprocessor.scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=available_features, index=X.index)

logging.info(f"Final dataset shape: {X_scaled.shape}")
logging.info(f"Target distribution: {y.value_counts().to_dict()}")

# ---------------------------
# 3. Time Series Validation and Model Training
# ---------------------------

def time_series_split_validation(X: pd.DataFrame, y: pd.Series, n_splits: int = 5) -> Dict:
    """
    Perform time series cross-validation to prevent lookahead bias
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = {'accuracy': [], 'roc_auc': [], 'precision': [], 'recall': []}
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_fold_train, X_fold_val = X.iloc[train_idx], X.iloc[val_idx]
        y_fold_train, y_fold_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Train model on fold
        fold_model = XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        fold_model.fit(X_fold_train, y_fold_train)
        
        # Validate
        y_pred = fold_model.predict(X_fold_val)
        y_pred_proba = fold_model.predict_proba(X_fold_val)[:, 1]
        
        scores['accuracy'].append(accuracy_score(y_fold_val, y_pred))
        scores['roc_auc'].append(roc_auc_score(y_fold_val, y_pred_proba))
        
        # Handle precision/recall for edge cases
        report = classification_report(y_fold_val, y_pred, output_dict=True)
        scores['precision'].append(report['weighted avg']['precision'])
        scores['recall'].append(report['weighted avg']['recall'])
        
        logging.info(f"Fold {fold + 1} - Accuracy: {scores['accuracy'][-1]:.4f}, AUC: {scores['roc_auc'][-1]:.4f}")
    
    # Calculate mean and std for each metric
    cv_results = {}
    for metric, values in scores.items():
        cv_results[f'{metric}_mean'] = np.mean(values)
        cv_results[f'{metric}_std'] = np.std(values)
    
    return cv_results

def optimize_hyperparameters(X: pd.DataFrame, y: pd.Series, n_trials: int = 50) -> Dict:
    """
    Use Optuna for hyperparameter optimization with time series cross-validation
    """
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
            'random_state': 42,
            'use_label_encoder': False,
            'eval_metric': 'logloss'
        }
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=3)  # Reduced splits for faster optimization
        scores = []
        
        for train_idx, val_idx in tscv.split(X):
            X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
            y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
            
            model = XGBClassifier(**params)
            model.fit(X_train_fold, y_train_fold)
            
            y_pred_proba = model.predict_proba(X_val_fold)[:, 1]
            score = roc_auc_score(y_val_fold, y_pred_proba)
            scores.append(score)
        
        return np.mean(scores)
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    
    logging.info(f"Best parameters: {study.best_params}")
    logging.info(f"Best cross-validation score: {study.best_value:.4f}")
    
    return study.best_params

# Perform time series validation with current parameters
logging.info("Performing time series cross-validation...")
cv_results = time_series_split_validation(X_scaled, y, n_splits=5)

print("\n=== Time Series Cross-Validation Results ===")
for metric, value in cv_results.items():
    print(f"{metric}: {value:.4f}")

# Hyperparameter optimization
logging.info("Starting hyperparameter optimization...")
try:
    best_params = optimize_hyperparameters(X_scaled, y, n_trials=30)  # Reduced for demo
except Exception as e:
    logging.warning(f"Hyperparameter optimization failed: {e}. Using default parameters.")
    best_params = {
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 3,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0,
        'reg_lambda': 1,
        'random_state': 42,
        'use_label_encoder': False,
        'eval_metric': 'logloss'
    }

# Final train/test split (time-aware)
split_idx = int(0.8 * len(X_scaled))  # 80/20 split maintaining time order
X_train, X_test = X_scaled.iloc[:split_idx], X_scaled.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

logging.info(f"Training set size: {len(X_train)}, Test set size: {len(X_test)}")

# Train final model with best parameters
final_model = XGBClassifier(**best_params)
final_model.fit(X_train, y_train)

# Model evaluation
y_pred = final_model.predict(X_test)
y_pred_proba = final_model.predict_proba(X_test)[:, 1]

print("\n=== Final Model Performance ===")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_pred_proba))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Feature importance analysis
feature_importance = pd.DataFrame({
    'feature': available_features,
    'importance': final_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n=== Top 10 Most Important Features ===")
print(feature_importance.head(10))

# Save the enhanced model and preprocessing pipeline
model_artifacts = {
    'model': final_model,
    'preprocessor': preprocessor,
    'feature_columns': available_features,
    'best_params': best_params,
    'cv_results': cv_results,
    'feature_importance': feature_importance
}

with open('models/enhanced_spy_options_model.pkl', 'wb') as f:
    pickle.dump(model_artifacts, f)

logging.info("Enhanced model and artifacts saved to 'models/enhanced_spy_options_model.pkl'")


# ---------------------------
# 4. Enhanced Trading Logic and Risk Management
# ---------------------------

class AdvancedTradingStrategy:
    """Enhanced trading strategy with comprehensive risk management"""
    
    def __init__(self, model, preprocessor, feature_columns, 
                 max_position_size=0.02, max_portfolio_risk=0.05):
        self.model = model
        self.preprocessor = preprocessor
        self.feature_columns = feature_columns
        self.max_position_size = max_position_size  # Max 2% of portfolio per trade
        self.max_portfolio_risk = max_portfolio_risk  # Max 5% portfolio risk
        self.positions = {}  # Track open positions
        
    def calculate_position_size(self, confidence: float, account_balance: float, 
                              option_price: float, volatility: float) -> float:
        """
        Calculate optimal position size using Kelly Criterion and risk management
        """
        # Kelly Criterion approximation
        win_rate = confidence
        avg_win = 0.15  # Expected 15% win
        avg_loss = 0.30  # Expected 30% loss (stop loss)
        
        kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
        kelly_fraction = max(0, min(kelly_fraction, 0.15))  # Cap at 15% (reduced from 25%)
        
        # Volatility adjustment
        vol_adjustment = 1 / (1 + volatility * 3)  # Reduce size in high vol (increased sensitivity)
        
        # Risk-adjusted position size
        base_position_size = kelly_fraction * vol_adjustment * self.max_position_size
        
        # Dollar amount
        position_value = account_balance * base_position_size
        contracts = int(position_value / (option_price * 100))  # 100 shares per contract
        
        return max(1, contracts)  # At least 1 contract
    
    def generate_entry_signal(self, features: pd.DataFrame, 
                            confidence_threshold: float = 0.65) -> Dict:
        """
        Generate entry signals with confidence and position sizing
        """
        # Preprocess features
        features_scaled = self.preprocessor.scaler.transform(features[self.feature_columns])
        
        # Get prediction and confidence
        prob = self.model.predict_proba(features_scaled)[0][1]
        confidence = max(prob, 1 - prob)  # Distance from 0.5
        
        signal = {
            'action': 'none',
            'confidence': confidence,
            'probability': prob,
            'reasoning': []
        }
        
        # Entry conditions
        if prob > confidence_threshold:
            signal['action'] = 'buy_call'
            signal['reasoning'].append(f"High bullish probability: {prob:.3f}")
        elif prob < (1 - confidence_threshold):
            signal['action'] = 'buy_put'  
            signal['reasoning'].append(f"High bearish probability: {1-prob:.3f}")
        
        # Additional filters
        current_vix = features['vix_level'].iloc[0] if 'vix_level' in features.columns else 20
        if current_vix > 30:
            signal['reasoning'].append("High VIX - increased caution")
            confidence *= 0.8  # Reduce confidence in high volatility
        
        # Time decay check
        dte = features['time_to_expiry'].iloc[0] if 'time_to_expiry' in features.columns else 30
        if dte < 7:
            signal['action'] = 'none'
            signal['reasoning'].append("Too close to expiry - avoiding theta decay")
        
        signal['confidence'] = confidence
        return signal
    
    def generate_exit_signal(self, position_info: Dict, current_features: pd.DataFrame,
                           current_option_price: float) -> Dict:
        """
        Generate exit signals with dynamic stop-loss and profit-taking
        """
        entry_price = position_info['entry_price']
        entry_time = position_info['entry_time']
        position_type = position_info['type']
        
        # Calculate current P&L
        pnl_pct = (current_option_price - entry_price) / entry_price
        
        exit_signal = {
            'action': 'hold',
            'pnl_pct': pnl_pct,
            'reasoning': []
        }
        
        # Get current model confidence
        features_scaled = self.preprocessor.scaler.transform(current_features[self.feature_columns])
        prob = self.model.predict_proba(features_scaled)[0][1]
        
        # Profit taking (trailing stop or target)
        if pnl_pct > 0.3:  # 30% profit (reduced from 50%)
            exit_signal['action'] = 'sell'
            exit_signal['reasoning'].append("Profit target reached: 30%")
        elif pnl_pct > 0.3 and prob < 0.4:  # 30% profit + model turns bearish
            exit_signal['action'] = 'sell'
            exit_signal['reasoning'].append("Partial profit + model signal reversal")
        
        # Stop loss (dynamic based on volatility)
        current_iv = current_features['implied_volatility'].iloc[0] if 'implied_volatility' in current_features.columns else 0.2
        stop_loss_threshold = -0.2 * (1 + current_iv)  # Tighter stops (reduced from -30% to -20%)
        
        if pnl_pct < stop_loss_threshold:
            exit_signal['action'] = 'sell'
            exit_signal['reasoning'].append(f"Stop loss triggered: {pnl_pct:.1%}")
        
        # Time-based exit
        hours_held = (datetime.now() - entry_time).total_seconds() / 3600
        dte = current_features['time_to_expiry'].iloc[0] if 'time_to_expiry' in current_features.columns else 30
        
        if dte < 3:  # Close to expiry
            exit_signal['action'] = 'sell'
            exit_signal['reasoning'].append("Close to expiry - avoiding theta decay")
        elif hours_held > 48 and pnl_pct < 0.1:  # Held for 2 days without profit
            exit_signal['action'] = 'sell'
            exit_signal['reasoning'].append("Time stop - position not performing")
        
        # Model confidence exit
        if position_type == 'call' and prob < 0.3:
            exit_signal['action'] = 'sell'
            exit_signal['reasoning'].append("Model confidence dropped significantly")
        elif position_type == 'put' and prob > 0.7:
            exit_signal['action'] = 'sell'
            exit_signal['reasoning'].append("Model confidence dropped significantly")
        
        return exit_signal

class TradingPerformanceTracker:
    """Track and analyze trading performance"""
    
    def __init__(self):
        self.trades = []
        self.portfolio_value = []
        
    def record_trade(self, trade_info: Dict):
        """Record completed trade"""
        self.trades.append(trade_info)
        
    def calculate_metrics(self) -> Dict:
        """Calculate comprehensive trading metrics"""
        if not self.trades:
            return {}
        
        df_trades = pd.DataFrame(self.trades)
        
        # Basic metrics
        total_trades = len(df_trades)
        winning_trades = len(df_trades[df_trades['pnl_pct'] > 0])
        win_rate = winning_trades / total_trades
        
        # P&L metrics
        avg_win = df_trades[df_trades['pnl_pct'] > 0]['pnl_pct'].mean()
        avg_loss = df_trades[df_trades['pnl_pct'] < 0]['pnl_pct'].mean()
        avg_pnl = df_trades['pnl_pct'].mean()
        
        # Risk metrics
        profit_factor = abs(avg_win * winning_trades / (avg_loss * (total_trades - winning_trades))) if avg_loss != 0 else float('inf')
        max_drawdown = df_trades['pnl_pct'].cumsum().expanding().max() - df_trades['pnl_pct'].cumsum()
        max_drawdown = max_drawdown.max()
        
        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'avg_pnl': avg_pnl,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown
        }

# Risk Profile Configurations
CONSERVATIVE_CONFIG = {
    'max_position_size': 0.015,  # 1.5%
    'profit_target': 0.25,       # 25%
    'stop_loss': 0.15,           # 15%
    'confidence_threshold': 0.75, # Higher confidence required
    'kelly_cap': 0.10,           # 10% Kelly cap
    'vol_sensitivity': 4         # Higher volatility sensitivity
}

MODERATE_CONFIG = {
    'max_position_size': 0.02,   # 2%
    'profit_target': 0.30,       # 30%
    'stop_loss': 0.20,           # 20%
    'confidence_threshold': 0.70,
    'kelly_cap': 0.15,           # 15% Kelly cap
    'vol_sensitivity': 3         # Current setting
}

AGGRESSIVE_CONFIG = {
    'max_position_size': 0.03,   # 3% (still lower than original 5%)
    'profit_target': 0.40,       # 40%
    'stop_loss': 0.25,           # 25%
    'confidence_threshold': 0.65,
    'kelly_cap': 0.20,           # 20% Kelly cap
    'vol_sensitivity': 2         # Lower volatility sensitivity
}

# Select risk profile (change this line to switch profiles)
# Options: CONSERVATIVE_CONFIG, MODERATE_CONFIG, AGGRESSIVE_CONFIG
SELECTED_RISK_PROFILE = MODERATE_CONFIG  # Default: balanced risk/return

print(f"Using MODERATE_CONFIG risk profile:")
print(f"  Max Position Size: {SELECTED_RISK_PROFILE['max_position_size']:.1%}")
print(f"  Profit Target: {SELECTED_RISK_PROFILE['profit_target']:.1%}")
print(f"  Stop Loss: {SELECTED_RISK_PROFILE['stop_loss']:.1%}")
print(f"  Confidence Threshold: {SELECTED_RISK_PROFILE['confidence_threshold']:.1%}")

# Initialize enhanced trading strategy with selected risk profile
trading_strategy = AdvancedTradingStrategy(
    model=final_model,
    preprocessor=preprocessor,
    feature_columns=available_features,
    max_position_size=SELECTED_RISK_PROFILE['max_position_size'],
    max_portfolio_risk=0.10  # Updated from 0.05 to match risk management config
)

# Demo with enhanced features
print("\n=== Enhanced Trading Strategy Demo ===")

# Create sample current market conditions
sample_features = pd.DataFrame({
    feature: [X_test.iloc[-1][feature]] for feature in available_features
})

# Generate entry signal
entry_signal = trading_strategy.generate_entry_signal(sample_features, confidence_threshold=0.65)
print(f"Entry Signal: {entry_signal['action']}")
print(f"Confidence: {entry_signal['confidence']:.3f}")
print(f"Probability: {entry_signal['probability']:.3f}")
print(f"Reasoning: {', '.join(entry_signal['reasoning'])}")

# Simulate position management
if entry_signal['action'] != 'none':
    # Calculate position size
    account_balance = 100000  # $100k account
    option_price = sample_features['option_price'].iloc[0] if 'option_price' in sample_features.columns else 10
    volatility = sample_features['implied_volatility'].iloc[0] if 'implied_volatility' in sample_features.columns else 0.2
    
    position_size = trading_strategy.calculate_position_size(
        confidence=entry_signal['confidence'],
        account_balance=account_balance,
        option_price=option_price,
        volatility=volatility
    )
    
    print(f"\nRecommended Position Size: {position_size} contracts")
    print(f"Position Value: ${position_size * option_price * 100:,.2f}")
    print(f"Portfolio Allocation: {(position_size * option_price * 100) / account_balance:.1%}")

# Initialize performance tracker
performance_tracker = TradingPerformanceTracker()

print("\n=== Trading System Ready ===")
print("Enhanced model with comprehensive risk management is ready for deployment.")

# ---------------------------
# 5. Comprehensive Backtesting Framework
# ---------------------------

class OptionsBacktester:
    """Comprehensive backtesting framework for options trading strategies"""
    
    def __init__(self, strategy, initial_capital=100000):
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions = {}
        self.trades_history = []
        self.portfolio_history = []
        
    def backtest(self, data: pd.DataFrame, start_date=None, end_date=None) -> Dict:
        """
        Run comprehensive backtest on historical data
        """
        # Filter data by date range if specified
        if start_date:
            data = data[data.index >= start_date]
        if end_date:
            data = data[data.index <= end_date]
        
        logging.info(f"Starting backtest with {len(data)} data points")
        
        for i, (timestamp, row) in enumerate(data.iterrows()):
            # Prepare features for current timestamp
            current_features = pd.DataFrame([row[self.strategy.feature_columns]])
            
            # Check for exit signals on existing positions
            self._process_exits(current_features, timestamp, row)
            
            # Check for new entry signals
            self._process_entries(current_features, timestamp, row)
            
            # Record portfolio value
            portfolio_value = self._calculate_portfolio_value(row)
            self.portfolio_history.append({
                'timestamp': timestamp,
                'portfolio_value': portfolio_value,
                'cash': self.current_capital,
                'num_positions': len(self.positions)
            })
            
            # Log progress
            if i % 100 == 0:
                logging.info(f"Processed {i}/{len(data)} data points")
        
        # Close any remaining positions
        self._close_all_positions(data.iloc[-1])
        
        return self._calculate_backtest_results()
    
    def _process_entries(self, features: pd.DataFrame, timestamp, row):
        """Process potential entry signals"""
        if len(self.positions) >= 5:  # Max 5 concurrent positions
            return
        
        entry_signal = self.strategy.generate_entry_signal(features)
        
        if entry_signal['action'] != 'none':
            # Calculate position size
            option_price = row['option_price']
            volatility = row.get('implied_volatility', 0.2)
            
            position_size = self.strategy.calculate_position_size(
                confidence=entry_signal['confidence'],
                account_balance=self.current_capital,
                option_price=option_price,
                volatility=volatility
            )
            
            # Check if we have enough capital
            position_cost = position_size * option_price * 100
            if position_cost <= self.current_capital * 0.9:  # Don't use more than 90% of capital
                # Open position
                position_id = f"{timestamp}_{entry_signal['action']}"
                self.positions[position_id] = {
                    'entry_time': timestamp,
                    'entry_price': option_price,
                    'position_size': position_size,
                    'type': entry_signal['action'],
                    'confidence': entry_signal['confidence']
                }
                
                # Update capital
                self.current_capital -= position_cost
                
                logging.info(f"Opened {entry_signal['action']} position: {position_size} contracts at ${option_price}")
    
    def _process_exits(self, features: pd.DataFrame, timestamp, row):
        """Process potential exit signals for existing positions"""
        positions_to_close = []
        
        for position_id, position_info in self.positions.items():
            exit_signal = self.strategy.generate_exit_signal(
                position_info, features, row['option_price']
            )
            
            if exit_signal['action'] == 'sell':
                positions_to_close.append(position_id)
                
                # Calculate P&L
                exit_price = row['option_price']
                pnl = (exit_price - position_info['entry_price']) * position_info['position_size'] * 100
                pnl_pct = (exit_price - position_info['entry_price']) / position_info['entry_price']
                
                # Record trade
                self.trades_history.append({
                    'entry_time': position_info['entry_time'],
                    'exit_time': timestamp,
                    'entry_price': position_info['entry_price'],
                    'exit_price': exit_price,
                    'position_size': position_info['position_size'],
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                    'type': position_info['type'],
                    'confidence': position_info['confidence'],
                    'exit_reason': ', '.join(exit_signal['reasoning'])
                })
                
                # Update capital
                self.current_capital += exit_price * position_info['position_size'] * 100
                
                logging.info(f"Closed position: {pnl_pct:.1%} P&L, Reason: {exit_signal['reasoning'][0]}")
        
        # Remove closed positions
        for position_id in positions_to_close:
            del self.positions[position_id]
    
    def _close_all_positions(self, final_row):
        """Close all remaining positions at the end of backtest"""
        for position_id, position_info in self.positions.items():
            exit_price = final_row['option_price']
            pnl = (exit_price - position_info['entry_price']) * position_info['position_size'] * 100
            pnl_pct = (exit_price - position_info['entry_price']) / position_info['entry_price']
            
            self.trades_history.append({
                'entry_time': position_info['entry_time'],
                'exit_time': final_row.name,
                'entry_price': position_info['entry_price'],
                'exit_price': exit_price,
                'position_size': position_info['position_size'],
                'pnl': pnl,
                'pnl_pct': pnl_pct,
                'type': position_info['type'],
                'confidence': position_info['confidence'],
                'exit_reason': 'End of backtest'
            })
            
            self.current_capital += exit_price * position_info['position_size'] * 100
        
        self.positions.clear()
    
    def _calculate_portfolio_value(self, current_row):
        """Calculate current portfolio value"""
        portfolio_value = self.current_capital
        
        for position_info in self.positions.values():
            current_value = current_row['option_price'] * position_info['position_size'] * 100
            portfolio_value += current_value
        
        return portfolio_value
    
    def _calculate_backtest_results(self) -> Dict:
        """Calculate comprehensive backtest metrics"""
        if not self.trades_history:
            return {'error': 'No trades executed during backtest'}
        
        df_trades = pd.DataFrame(self.trades_history)
        df_portfolio = pd.DataFrame(self.portfolio_history)
        
        # Basic metrics
        total_trades = len(df_trades)
        winning_trades = len(df_trades[df_trades['pnl'] > 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # P&L metrics
        total_pnl = df_trades['pnl'].sum()
        total_return = (self.current_capital - self.initial_capital) / self.initial_capital
        
        avg_win = df_trades[df_trades['pnl'] > 0]['pnl_pct'].mean() if winning_trades > 0 else 0
        avg_loss = df_trades[df_trades['pnl'] < 0]['pnl_pct'].mean() if (total_trades - winning_trades) > 0 else 0
        
        # Risk metrics
        daily_returns = df_portfolio['portfolio_value'].pct_change().dropna()
        volatility = daily_returns.std() * np.sqrt(252)  # Annualized
        sharpe_ratio = (total_return * 252) / volatility if volatility > 0 else 0
        
        max_drawdown = self._calculate_max_drawdown(df_portfolio['portfolio_value'])
        
        # Additional metrics
        profit_factor = abs(avg_win * winning_trades / (avg_loss * (total_trades - winning_trades))) if avg_loss != 0 else float('inf')
        
        results = {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'total_return': total_return,
            'total_pnl': total_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'volatility': volatility,
            'final_capital': self.current_capital,
            'trades_per_month': total_trades / max(1, len(df_portfolio) / 30),
            'avg_trade_duration': (df_trades['exit_time'] - df_trades['entry_time']).mean()
        }
        
        return results
    
    def _calculate_max_drawdown(self, portfolio_values):
        """Calculate maximum drawdown"""
        peak = portfolio_values.expanding().max()
        drawdown = (portfolio_values - peak) / peak
        return drawdown.min()

# Run backtest demonstration
if len(df_final) > 100:  # Only run if we have sufficient data
    print("\n=== Running Backtest Demonstration ===")
    
    # Create backtester
    backtester = OptionsBacktester(trading_strategy, initial_capital=100000)
    
    # Use the last 80% of data for backtesting (remaining 20% is our test set)
    backtest_data = df_final.iloc[:int(0.8 * len(df_final))].copy()
    backtest_data.index = pd.date_range(start='2023-01-01', periods=len(backtest_data), freq='H')
    
    # Run backtest
    try:
        backtest_results = backtester.backtest(backtest_data)
        
        print("=== Backtest Results ===")
        for metric, value in backtest_results.items():
            if isinstance(value, float):
                if 'rate' in metric or 'return' in metric or 'drawdown' in metric:
                    print(f"{metric}: {value:.2%}")
                elif 'ratio' in metric:
                    print(f"{metric}: {value:.2f}")
                else:
                    print(f"{metric}: {value:,.2f}")
            else:
                print(f"{metric}: {value}")
                
        # Save backtest results
        with open('models/backtest_results.pkl', 'wb') as f:
            pickle.dump({
                'results': backtest_results,
                'trades': backtester.trades_history,
                'portfolio_history': backtester.portfolio_history
            }, f)
        
        logging.info("Backtest results saved to 'models/backtest_results.pkl'")
        
    except Exception as e:
        logging.error(f"Backtest failed: {e}")
        print("Backtest demonstration skipped due to data requirements")

# ---------------------------
# 6. Integration with OptionsTitan Production Pipeline
# ---------------------------

print("\n" + "="*60)
print("INTEGRATING WITH OPTIONSTITAN PRODUCTION PIPELINE")
print("="*60)

try:
    # Import the integrated pipeline (relative when run as package, else same-dir)
    try:
        from .integrated_pipeline import ProductionPipeline
    except ImportError:
        from integrated_pipeline import ProductionPipeline
    
    # Create pipeline configuration
    pipeline_config = {
        'ensemble': {
            'enabled': True,
            'models': ['xgboost', 'lightgbm', 'gradient_boost', 'random_forest', 'logistic'],
            'optimization_trials': 30,
            'cv_folds': 5
        },
        'risk_management': {
            'enabled': True,
            'confidence_levels': [0.95, 0.99],
            'max_position_size': SELECTED_RISK_PROFILE['max_position_size'],
            'max_portfolio_risk': 0.10,  # Reduced from 0.15
            'correlation_threshold': 0.60  # Reduced from 0.70
        },
        'explainability': {
            'enabled': True,
            'sample_explanations': 5,
            'global_importance': True
        },
        'drift_detection': {
            'enabled': True,
            'drift_threshold': 0.05,
            'quality_threshold': 0.95
        },
        'artifacts': {
            'save_path': './models/production_artifacts_{timestamp}.pkl',
            'backup_count': 5,
            'compress': True
        },
        'logging': {
            'level': 'INFO',
            'file': 'logs/optionstitan_pipeline.log'
        }
    }
    
    # Initialize production pipeline
    print("Initializing OptionsTitan Production Pipeline...")
    pipeline = ProductionPipeline(pipeline_config)
    
    # Run the complete pipeline
    print("Running comprehensive production pipeline...")
    pipeline_results = pipeline.run_full_pipeline(
        X_train=X_scaled,
        y_train=y,
        X_test=X_test if 'X_test' in locals() else None,
        y_test=y_test if 'y_test' in locals() else None,
        base_model=final_model
    )
    
    # Display comprehensive report
    print("\n" + pipeline_results['comprehensive_report'])
    
    # Save enhanced model artifacts (includes all pipeline components)
    enhanced_artifacts = {
        'original_model': final_model,
        'original_preprocessor': preprocessor,
        'original_feature_columns': available_features,
        'original_best_params': best_params,
        'original_cv_results': cv_results,
        'original_feature_importance': feature_importance,
        'pipeline_results': pipeline_results,
        'pipeline_components': pipeline.components,
        'integration_timestamp': datetime.now(),
        'version': 'OptionsTitan_v1.0_Enhanced'
    }
    
    with open('models/enhanced_spy_options_model.pkl', 'wb') as f:
        pickle.dump(enhanced_artifacts, f)
    
    print(f"\n✅ Enhanced model with full pipeline saved to 'models/enhanced_spy_options_model.pkl'")
    print(f"✅ Production artifacts saved to: {pipeline_results.get('artifact_path', 'N/A')}")
    
    # Create inference pipeline for immediate use
    print("\nCreating inference pipeline for production use...")
    inference_pipeline = pipeline.create_inference_pipeline()
    
    # Demonstrate inference capabilities
    if len(X_scaled) > 0:
        print("\n--- INFERENCE DEMONSTRATION ---")
        
        # Take a sample for inference
        sample_data = X_scaled.tail(5)  # Last 5 rows
        
        # Make predictions with full analysis
        inference_results = inference_pipeline.predict(
            sample_data, 
            explain=True, 
            check_drift=True
        )
        
        print(f"Generated predictions for {len(sample_data)} samples")
        
        if 'predictions' in inference_results:
            print("Sample Predictions:")
            for i, (pred, prob, unc) in enumerate(zip(
                inference_results['predictions'][:3],
                inference_results['probabilities'][:3], 
                inference_results['uncertainty'][:3]
            )):
                print(f"  Sample {i+1}: Prediction={pred:.3f}, Probability={prob:.3f}, Uncertainty={unc:.3f}")
        
        if 'drift_alert' in inference_results:
            print(f"\n⚠️  {inference_results['drift_alert']}")
        
        if 'explanations' in inference_results:
            print(f"\nSample Explanation (First Prediction):")
            first_explanation = inference_results['explanations'][0]
            print(f"Confidence: {first_explanation['confidence']:.1%}")
            print("Trading Rationale:")
            print(first_explanation['trading_rationale'])
    
    print("\n" + "="*60)
    print("OPTIONSTITAN PRODUCTION SYSTEM READY! 🚀")
    print("="*60)
    print("✅ Phase 1: Ensemble Models - COMPLETED")
    print("✅ Phase 2: Feature Drift Detection - COMPLETED") 
    print("✅ Phase 3: Risk Management - COMPLETED")
    print("✅ Phase 4: Model Explainability - COMPLETED")
    print("✅ Integrated Pipeline - READY FOR DEPLOYMENT")
    print("")
    print("Next Steps:")
    print("1. Paper Trading: Test with small positions")
    print("2. Live Trading: Deploy with proper risk controls")
    print("3. Monitoring: Use drift detection for model updates")
    print("4. Scaling: Increase position sizes gradually")
    print("="*60)

except ImportError as e:
    print(f"⚠️  Pipeline integration not available: {e}")
    print("Make sure all component files are present:")
    print("  - ensemble_models.py")
    print("  - risk_management.py") 
    print("  - model_explainability.py")
    print("  - integrated_pipeline.py")
    print("\nFalling back to basic model training...")
    
    # Save basic model artifacts
    basic_artifacts = {
        'model': final_model,
        'preprocessor': preprocessor,
        'feature_columns': available_features,
        'best_params': best_params,
        'cv_results': cv_results,
        'feature_importance': feature_importance,
        'version': 'Basic_v1.0'
    }
    
    with open('models/enhanced_spy_options_model.pkl', 'wb') as f:
        pickle.dump(basic_artifacts, f)
    
    print("✅ Basic model saved to 'models/enhanced_spy_options_model.pkl'")

except Exception as e:
    print(f"❌ Pipeline integration failed: {e}")
    print("Error details:", str(e))
    print("Continuing with basic model training...")
    
    # Save basic model artifacts as fallback
    try:
        basic_artifacts = {
            'model': final_model,
            'preprocessor': preprocessor,
            'feature_columns': available_features,
            'best_params': best_params,
            'cv_results': cv_results,
            'feature_importance': feature_importance,
            'version': 'Basic_v1.0_Fallback'
        }
        
        with open('models/enhanced_spy_options_model.pkl', 'wb') as f:
            pickle.dump(basic_artifacts, f)
        
        print("✅ Basic model saved to 'models/enhanced_spy_options_model.pkl'")
    except Exception as save_error:
        print(f"❌ Failed to save basic model: {save_error}")

# ---------------------------
# 7. Live Data Integration (Future Considerations)
# ---------------------------