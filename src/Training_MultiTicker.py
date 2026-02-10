"""
Multi-Ticker Options Trading Model Training

Extends Training.py to handle multi-year, multi-ticker datasets with:
- Walk-forward validation (2019-2020 -> 2021, 2019-2021 -> 2022, etc.)
- Ticker/sector categorical features
- VIX regime balancing for COVID period
- Universal model across all tickers

Data source: FREE philippdubach/options-data (via remote_query module)
Schema: Same as Training.py (10 required columns)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from sklearn.preprocessing import RobustScaler
import pickle
import warnings
from datetime import datetime
import logging
from typing import Dict, Tuple, List

# Import existing Training.py components
import sys
sys.path.append(str(Path(__file__).parent))
from Training import (
    DataPreprocessor, FeatureEngineer, create_target_variable,
    AdvancedTradingStrategy, TradingPerformanceTracker,
    MODERATE_CONFIG
)

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# MULTI-TICKER DATA LOADING
# ============================================================================

class MultiTickerDataLoader:
    """Load and combine multi-ticker datasets"""
    
    def __init__(self, data_dir: str = 'data/processed_csv'):
        self.data_dir = Path(data_dir)
        
        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"Data directory not found: {data_dir}\n\n"
                "To collect training data using FREE philippdubach dataset:\n"
                "  1. Run validation: python test_free_data_migration.py\n"
                "  2. Query data: See docs/DATA_COLLECTION_GUIDE.md\n"
                "  3. Or use remote_query module:\n"
                "     from src.data_collection.remote_query import batch_query_tickers\n\n"
                "See README_DATA_COLLECTION.md for complete guide."
            )
    
    def load_single_ticker(self, ticker: str) -> pd.DataFrame:
        """Load data for a single ticker"""
        # Find CSV files matching ticker
        pattern = f"{ticker.lower()}_*_options.csv"
        files = list(self.data_dir.glob(pattern))
        
        if not files:
            logger.warning(f"No data files found for {ticker}")
            return pd.DataFrame()
        
        # Load and combine all files for this ticker
        dfs = []
        for file in files:
            df = pd.read_csv(file)
            df['ticker'] = ticker
            dfs.append(df)
        
        combined = pd.concat(dfs, ignore_index=True)
        logger.info(f"Loaded {len(combined)} rows for {ticker}")
        
        return combined
    
    def load_all_tickers(self, tickers: List[str] = None) -> pd.DataFrame:
        """Load data for all tickers"""
        if tickers is None:
            # Auto-detect tickers from CSV files
            csv_files = list(self.data_dir.glob('*_options.csv'))
            tickers = list(set([
                f.name.split('_')[0].upper()
                for f in csv_files
            ]))
            logger.info(f"Auto-detected {len(tickers)} tickers: {tickers}")
        
        all_data = []
        
        for ticker in tickers:
            df = self.load_single_ticker(ticker)
            if not df.empty:
                all_data.append(df)
        
        if not all_data:
            raise ValueError("No data loaded for any ticker")
        
        combined = pd.concat(all_data, ignore_index=True)
        
        logger.info(f"\nLoaded multi-ticker dataset:")
        logger.info(f"  Total rows: {len(combined)}")
        logger.info(f"  Tickers: {len(tickers)}")
        logger.info(f"  Date range: {combined['timestamp'].min()} to {combined['timestamp'].max()}")
        
        return combined
    
    def filter_by_date_range(self, 
                            df: pd.DataFrame, 
                            start_date: str, 
                            end_date: str) -> pd.DataFrame:
        """Filter dataset to date range"""
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        mask = (df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)
        filtered = df[mask]
        
        logger.info(f"Filtered to {start_date} - {end_date}: {len(filtered)} rows")
        
        return filtered


# ============================================================================
# MULTI-TICKER FEATURE ENGINEERING
# ============================================================================

class MultiTickerFeatureEngineer:
    """Add ticker-specific and sector features"""
    
    def __init__(self):
        # Sector mapping (from config.py)
        self.sector_map = {
            # ETFs
            'SPY': 'ETF', 'QQQ': 'ETF', 'IWM': 'ETF', 'DIA': 'ETF',
            'XLF': 'ETF', 'XLK': 'ETF', 'XLE': 'ETF',
            
            # Tech
            'AAPL': 'Tech', 'MSFT': 'Tech', 'NVDA': 'Tech', 'AMZN': 'Tech',
            'GOOGL': 'Tech', 'META': 'Tech', 'TSLA': 'Tech', 'AMD': 'Tech',
            'NFLX': 'Tech', 'ORCL': 'Tech',
            
            # Financials
            'JPM': 'Financial', 'BAC': 'Financial', 'GS': 'Financial',
            'MS': 'Financial', 'WFC': 'Financial', 'SCHW': 'Financial',
            
            # Healthcare
            'UNH': 'Healthcare', 'JNJ': 'Healthcare', 'PFE': 'Healthcare',
            'ABBV': 'Healthcare', 'LLY': 'Healthcare',
            
            # Energy
            'XOM': 'Energy', 'CVX': 'Energy', 'COP': 'Energy', 'SLB': 'Energy',
            
            # Consumer
            'WMT': 'Consumer', 'COST': 'Consumer', 'TGT': 'Consumer',
            'DIS': 'Consumer', 'NKE': 'Consumer',
            
            # High Vol
            'GME': 'High_Vol', 'AMC': 'High_Vol', 'PLTR': 'High_Vol',
            'COIN': 'High_Vol', 'ROKU': 'High_Vol', 'SNAP': 'High_Vol',
            'DKNG': 'High_Vol', 'MARA': 'High_Vol',
            
            # Low Vol
            'KO': 'Low_Vol', 'PG': 'Low_Vol', 'PEP': 'Low_Vol',
            'MCD': 'Low_Vol', 'V': 'Low_Vol',
        }
    
    def add_ticker_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add ticker categorical encoding and sector features"""
        df = df.copy()
        
        # Ticker categorical encoding
        df['ticker_encoded'] = df['ticker'].astype('category').cat.codes
        
        # Sector encoding
        df['sector'] = df['ticker'].map(self.sector_map).fillna('Other')
        df['sector_encoded'] = df['sector'].astype('category').cat.codes
        
        # Ticker-level statistics (for normalization)
        ticker_stats = df.groupby('ticker').agg({
            'implied_volatility': ['mean', 'std'],
            'volume': 'median',
            'option_price': 'median'
        }).reset_index()
        
        ticker_stats.columns = ['ticker', 'ticker_iv_mean', 'ticker_iv_std',
                               'ticker_volume_median', 'ticker_price_median']
        
        df = df.merge(ticker_stats, on='ticker', how='left')
        
        # Normalized features (relative to ticker baseline)
        df['iv_vs_ticker_avg'] = (
            (df['implied_volatility'] - df['ticker_iv_mean']) / 
            df['ticker_iv_std'].clip(lower=0.01)
        )
        
        logger.info(f"Added ticker features: {df['ticker'].nunique()} unique tickers")
        
        return df
    
    def add_market_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market regime classification (COVID, rate hikes, etc.)"""
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Define regime periods
        regimes = {
            'pre_covid': ('2019-01-01', '2020-02-29'),
            'covid_crash': ('2020-03-01', '2020-04-30'),
            'covid_recovery': ('2020-05-01', '2021-12-31'),
            'rate_hikes': ('2022-01-01', '2023-12-31'),
            'current': ('2024-01-01', '2026-12-31'),
        }
        
        def classify_regime(timestamp):
            for regime_name, (start, end) in regimes.items():
                if pd.Timestamp(start) <= timestamp <= pd.Timestamp(end):
                    return regime_name
            return 'other'
        
        df['market_regime'] = df['timestamp'].apply(classify_regime)
        df['regime_encoded'] = df['market_regime'].astype('category').cat.codes
        
        # VIX regime bucketing
        if 'vix_level' in df.columns:
            df['vix_regime'] = pd.cut(
                df['vix_level'],
                bins=[0, 15, 25, 35, 100],
                labels=['low_vol', 'normal', 'elevated', 'crisis']
            )
            df['vix_regime_encoded'] = df['vix_regime'].cat.codes
        
        logger.info(f"Added regime features: {df['market_regime'].value_counts().to_dict()}")
        
        return df


# ============================================================================
# WALK-FORWARD VALIDATION
# ============================================================================

class WalkForwardValidator:
    """
    Implement walk-forward validation strategy
    
    Validation folds:
    - Train 2019-2020 → Validate 2021
    - Train 2019-2021 → Validate 2022
    - Train 2019-2022 → Validate 2023
    - Train 2019-2023 → Test 2024
    """
    
    def __init__(self):
        self.folds = [
            {'train': ('2019-01-01', '2020-12-31'), 'val': ('2021-01-01', '2021-12-31')},
            {'train': ('2019-01-01', '2021-12-31'), 'val': ('2022-01-01', '2022-12-31')},
            {'train': ('2019-01-01', '2022-12-31'), 'val': ('2023-01-01', '2023-12-31')},
            {'train': ('2019-01-01', '2023-12-31'), 'test': ('2024-01-01', '2024-12-31')},
        ]
    
    def get_fold(self, df: pd.DataFrame, fold_idx: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Get train/val split for a fold"""
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        fold = self.folds[fold_idx]
        
        # Train set
        train_start, train_end = fold['train']
        train_mask = (df['timestamp'] >= train_start) & (df['timestamp'] <= train_end)
        train_df = df[train_mask]
        
        # Val/test set
        val_key = 'test' if 'test' in fold else 'val'
        val_start, val_end = fold[val_key]
        val_mask = (df['timestamp'] >= val_start) & (df['timestamp'] <= val_end)
        val_df = df[val_mask]
        
        logger.info(f"\nFold {fold_idx + 1}:")
        logger.info(f"  Train: {train_start} to {train_end} ({len(train_df)} rows)")
        logger.info(f"  Val:   {val_start} to {val_end} ({len(val_df)} rows)")
        
        return train_df, val_df
    
    def run_validation(self,
                      df: pd.DataFrame,
                      feature_columns: List[str],
                      model_params: Dict) -> Dict:
        """Run walk-forward validation across all folds"""
        results = {}
        
        for fold_idx in range(len(self.folds)):
            train_df, val_df = self.get_fold(df, fold_idx)
            
            if train_df.empty or val_df.empty:
                logger.warning(f"Skipping fold {fold_idx + 1}: empty data")
                continue
            
            # Prepare features
            X_train = train_df[feature_columns]
            y_train = train_df['target']
            X_val = val_df[feature_columns]
            y_val = val_df['target']
            
            # Train model
            model = XGBClassifier(**model_params)
            model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = model.predict(X_val)
            y_pred_proba = model.predict_proba(X_val)[:, 1]
            
            fold_results = {
                'accuracy': accuracy_score(y_val, y_pred),
                'roc_auc': roc_auc_score(y_val, y_pred_proba),
                'classification_report': classification_report(y_val, y_pred, output_dict=True),
            }
            
            results[f'fold_{fold_idx + 1}'] = fold_results
            
            logger.info(f"  Accuracy: {fold_results['accuracy']:.4f}")
            logger.info(f"  ROC AUC:  {fold_results['roc_auc']:.4f}")
        
        # Calculate average performance
        avg_accuracy = np.mean([r['accuracy'] for r in results.values()])
        avg_auc = np.mean([r['roc_auc'] for r in results.values()])
        
        results['average'] = {
            'accuracy': avg_accuracy,
            'roc_auc': avg_auc
        }
        
        logger.info(f"\nAverage Performance:")
        logger.info(f"  Accuracy: {avg_accuracy:.4f}")
        logger.info(f"  ROC AUC:  {avg_auc:.4f}")
        
        return results


# ============================================================================
# ============================================================================
# SAMPLE DATA FETCH (FREE philippdubach dataset)
# ============================================================================

def fetch_sample_data(
    data_dir: str = 'data/processed_csv',
    tickers: List[str] = None,
    date_min: str = '2024-10-01',
    date_max: str = '2024-12-31',
) -> bool:
    """
    Fetch a small sample of options data using FREE remote_query and save to data_dir.
    Creates CSVs in the format expected by MultiTickerDataLoader.
    """
    import sys
    try:
        from src.data_collection.remote_query import query_remote_parquet, query_underlying_prices
        from src.data_collection.data_normalizer import MultiYearNormalizer
    except ImportError:
        try:
            from .data_collection.remote_query import query_remote_parquet, query_underlying_prices
            from .data_collection.data_normalizer import MultiYearNormalizer
        except ImportError:
            # When run as script, add project root to path
            _root = Path(__file__).resolve().parent.parent
            if str(_root) not in sys.path:
                sys.path.insert(0, str(_root))
            try:
                from src.data_collection.remote_query import query_remote_parquet, query_underlying_prices
                from src.data_collection.data_normalizer import MultiYearNormalizer
            except ImportError as e:
                logger.error("Cannot import remote_query or data_normalizer: %s. Run from project root: python -m src.Training_MultiTicker --fetch-sample", e)
                return False

    tickers = tickers or ['spy', 'aapl']
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    normalizer = MultiYearNormalizer()

    filters = {
        'date_min': date_min,
        'date_max': date_max,
        'dte_min': 30,
        'dte_max': 60,
        'min_volume': 1,
        'min_open_interest': 100,
    }

    for ticker in tickers:
        ticker_lower = ticker.lower()
        print(f"  Fetching {ticker.upper()} options...")
        raw = query_remote_parquet(ticker_lower, filters)
        if raw is None or raw.empty:
            logger.warning(f"  No data for {ticker.upper()}, skipping")
            continue
        if 'expiration' in raw.columns and 'expiration_date' not in raw.columns:
            raw = raw.copy()
            raw['expiration_date'] = pd.to_datetime(raw['expiration'])
        print(f"  Fetching {ticker.upper()} underlying prices...")
        underlying = query_underlying_prices(ticker_lower, date_min=date_min, date_max=date_max)
        if underlying is not None and not underlying.empty:
            underlying = underlying.rename(columns={'close': 'Close', 'volume': 'Volume'})
            if 'date' not in underlying.columns and 'Date' in underlying.columns:
                underlying = underlying.rename(columns={'Date': 'date'})
        else:
            underlying = None
        norm = normalizer.normalize_dataset(raw, ticker.upper(), underlying)
        if norm is None or norm.empty:
            logger.warning(f"  Normalization produced no rows for {ticker.upper()}, skipping")
            continue
        out_file = data_path / f"{ticker_lower}_{date_min[:4]}_q4_options.csv"
        norm.to_csv(out_file, index=False, float_format='%.6f')
        print(f"  Saved {len(norm):,} rows to {out_file}")
    return True


# MAIN TRAINING PIPELINE
# ============================================================================

def main(data_dir: str = 'data/processed_csv', fetch_sample: bool = False):
    """Main multi-ticker training pipeline"""
    
    print("="*70)
    print("MULTI-TICKER OPTIONS TRADING MODEL TRAINING")
    print("="*70)
    
    data_path = Path(data_dir)
    if not data_path.exists() and fetch_sample:
        print("\n[0/7] Fetching sample data (FREE philippdubach dataset)...")
        if not fetch_sample_data(data_dir=data_dir):
            print("\n❌ Sample fetch failed. See errors above.")
            return
        print("✓ Sample data ready.\n")
    if not data_path.exists():
        print("\n[1/7] Loading multi-ticker dataset...")
        try:
            loader = MultiTickerDataLoader(data_dir=data_dir)
        except FileNotFoundError as e:
            print(f"\n❌ Error: {e}")
            print("\nTo fetch sample data automatically, run:")
            print("  python -m src.Training_MultiTicker --fetch-sample")
            return
        df = loader.load_all_tickers()
    else:
        print("\n[1/7] Loading multi-ticker dataset...")
        loader = MultiTickerDataLoader(data_dir=data_dir)
        try:
            df = loader.load_all_tickers()
        except FileNotFoundError as e:
            print(f"\n❌ Error: {e}")
            return
    
    print(f"✓ Loaded {len(df)} rows across {df['ticker'].nunique()} tickers")
    
    # Step 2: Data preprocessing
    print("\n[2/7] Preprocessing data...")
    preprocessor = DataPreprocessor()
    
    df_clean = preprocessor.handle_missing_values(df)
    numeric_columns = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    df_clean = preprocessor.detect_outliers(df_clean, numeric_columns, method='iqr')
    
    print(f"✓ Cleaned dataset: {len(df_clean)} rows")
    
    # Step 3: Feature engineering
    print("\n[3/7] Engineering features...")
    
    # Basic features from Training.py
    feature_engineer = FeatureEngineer()
    df_features = feature_engineer.calculate_greeks_approximation(df_clean)
    df_features = feature_engineer.create_time_features(df_features)
    df_features = feature_engineer.create_market_regime_features(df_features)
    
    # Multi-ticker features
    mt_engineer = MultiTickerFeatureEngineer()
    df_features = mt_engineer.add_ticker_features(df_features)
    df_features = mt_engineer.add_market_regime_features(df_features)
    
    print(f"✓ Engineered features")
    
    # Step 4: Create target variable
    print("\n[4/7] Creating target variable...")
    df_final = create_target_variable(
        df_features,
        profit_threshold=0.05,
        time_horizon=1,
        transaction_cost=0.005,
        risk_adjusted=True
    )
    
    print(f"✓ Target created: {df_final['target'].value_counts().to_dict()}")
    
    # Step 5: Feature selection
    print("\n[5/7] Selecting features...")
    
    feature_columns = [
        # Original features
        'price', 'option_price', 'strike_distance', 'time_to_expiry',
        'volume', 'implied_volatility', 'vix_level', 'spy_return_5min', 'rsi',
        
        # Greeks
        'delta_approx', 'gamma_approx', 'theta_approx',
        
        # Time features
        'hour', 'day_of_week', 'is_market_open', 'time_to_close',
        
        # Market regime
        'vix_regime_encoded', 'iv_percentile', 'iv_rank',
        
        # Multi-ticker features
        'ticker_encoded', 'sector_encoded', 'regime_encoded',
        'iv_vs_ticker_avg', 'ticker_iv_mean', 'ticker_volume_median',
    ]
    
    # Filter to available features
    available_features = [f for f in feature_columns if f in df_final.columns]
    
    print(f"✓ Selected {len(available_features)} features")
    
    # Step 6: Walk-forward validation
    print("\n[6/7] Running walk-forward validation...")
    
    validator = WalkForwardValidator()
    
    model_params = {
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 5,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'use_label_encoder': False,
        'eval_metric': 'logloss'
    }
    
    validation_results = validator.run_validation(
        df_final,
        available_features,
        model_params
    )
    
    print(f"✓ Validation complete")
    
    # Step 7: Train final model on all data
    print("\n[7/7] Training final model...")
    
    X = df_final[available_features]
    y = df_final['target']
    
    # Scale features
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=available_features, index=X.index)
    
    # Train final model
    final_model = XGBClassifier(**model_params)
    final_model.fit(X_scaled, y)
    
    print(f"✓ Final model trained")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': available_features,
        'importance': final_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Features:")
    print(feature_importance.head(10).to_string(index=False))
    
    # Save model artifacts
    print("\n[8/7] Saving model...")
    
    model_artifacts = {
        'model': final_model,
        'preprocessor': preprocessor,
        'scaler': scaler,
        'feature_columns': available_features,
        'model_params': model_params,
        'validation_results': validation_results,
        'feature_importance': feature_importance,
        'training_date': datetime.now(),
        'ticker_count': df['ticker'].nunique(),
        'total_rows': len(df_final),
        'version': 'MultiTicker_v1.0'
    }
    
    # Create models directory
    models_dir = Path('models')
    models_dir.mkdir(exist_ok=True)
    
    model_path = models_dir / 'multi_ticker_options_model.pkl'
    
    with open(model_path, 'wb') as f:
        pickle.dump(model_artifacts, f)
    
    print(f"✓ Model saved to: {model_path}")
    
    # Summary
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"Model: {model_path}")
    print(f"Tickers: {df['ticker'].nunique()}")
    print(f"Training samples: {len(df_final)}")
    print(f"Features: {len(available_features)}")
    print(f"Validation accuracy: {validation_results['average']['accuracy']:.4f}")
    print(f"Validation ROC AUC: {validation_results['average']['roc_auc']:.4f}")
    print("="*70)
    
    print("\nNext steps:")
    print("1. Backtest the model using integrated_pipeline.py")
    print("2. Integrate with GUI for live predictions")
    print("3. Collect more data for continuous improvement")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Multi-ticker options model training')
    parser.add_argument('--data-dir', default='data/processed_csv', help='Directory with ticker CSVs (default: data/processed_csv)')
    parser.add_argument('--fetch-sample', action='store_true', help='If data dir is missing, fetch sample data from FREE philippdubach dataset (SPY + AAPL, 2024 Q4)')
    args = parser.parse_args()
    main(data_dir=args.data_dir, fetch_sample=args.fetch_sample)
