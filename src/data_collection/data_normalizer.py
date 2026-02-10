"""
Data Normalizer

Normalizes raw options data to Training.py schema with multi-year
preprocessing including:
- Stock split adjustments
- Volume normalization (log1p, rolling median)
- IV percentile ranks
- Price returns instead of absolute levels
- Technical indicator calculation (RSI, returns)

Output schema matches Training.py's expected 10 columns.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
import yfinance as yf

# Try to import py_vollib for IV calculation
try:
    from py_vollib.black_scholes import black_scholes as bs
    from py_vollib.black_scholes.implied_volatility import implied_volatility as iv_calc
    VOLLIB_AVAILABLE = True
except ImportError:
    VOLLIB_AVAILABLE = False
    logging.warning("py_vollib not available. IV calculation will use approximations.")

from .config import REQUIRED_COLUMNS, OPTIONAL_COLUMNS, QUALITY_THRESHOLDS

logger = logging.getLogger(__name__)


class MultiYearNormalizer:
    """
    Normalize and transform multi-year options data to Training.py schema
    
    Handles:
    - Stock splits (AAPL, TSLA, NVDA, etc.)
    - Volume scaling across years
    - IV calculation/approximation
    - Technical indicators (RSI)
    - VIX level integration
    - Returns calculation
    """
    
    def __init__(self):
        """Initialize normalizer with caching for supplementary data"""
        self.vix_cache = None
        self.stock_splits_cache = {}
    
    def normalize_dataset(self,
                         raw_df: pd.DataFrame,
                         ticker: str,
                         underlying_prices: pd.DataFrame = None) -> pd.DataFrame:
        """
        Transform raw options data + underlying prices to Training.py schema
        
        Args:
            raw_df: Raw option data
                Expected columns: date, close (option), volume, bid, ask, strike, expiration
            ticker: Stock ticker symbol
            underlying_prices: DataFrame with underlying OHLCV data
                Required columns: date, Close, Volume
        
        Returns:
            DataFrame with Training.py schema (10 required columns)
        """
        if raw_df.empty:
            logger.warning(f"Empty dataset for {ticker}")
            return pd.DataFrame()
        
        logger.info(f"Normalizing {len(raw_df)} rows for {ticker}")
        
        df = raw_df.copy()
        
        # Ensure date column
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        
        # Step 1: Handle stock splits
        df = self.handle_stock_splits(df, ticker)
        
        # Step 2: Merge with underlying prices
        if underlying_prices is not None:
            df = self._merge_underlying_prices(df, underlying_prices)
        
        # Step 3: Calculate strike_distance
        if 'strike' in df.columns and 'price' in df.columns:
            df['strike_distance'] = df['strike'] - df['price']
        
        # Step 4: Calculate time_to_expiry (DTE)
        if 'expiration_date' in df.columns:
            df['expiration_date'] = pd.to_datetime(df['expiration_date'])
            df['time_to_expiry'] = (df['expiration_date'] - df['date']).dt.days
        
        # Step 5: Map option_price from 'close' or 'mark'
        if 'close' in df.columns:
            df['option_price'] = df['close']
        elif 'mark' in df.columns:
            df['option_price'] = df['mark']
        elif 'bid' in df.columns and 'ask' in df.columns:
            # Use mid price if close not available
            df['option_price'] = (df['bid'] + df['ask']) / 2.0
        
        # Step 6: Normalize volume
        df = self._normalize_volume(df)
        
        # Step 7: Calculate/estimate implied_volatility
        df = self.calculate_implied_volatility(df, ticker)
        
        # Step 8: Add VIX level
        df = self._add_vix_level(df)
        
        # Step 9: Calculate returns (proxy for spy_return_5min)
        df = self._calculate_returns(df)
        
        # Step 10: Calculate RSI
        if underlying_prices is not None:
            df = self._calculate_rsi(df, underlying_prices)
        
        # Step 11: Create timestamp column
        df = self._create_timestamp(df)
        
        # Step 12: Filter to required columns
        df = self._filter_to_schema(df, ticker)
        
        # Step 13: Quality checks
        df = self._apply_quality_filters(df)
        
        logger.info(f"Normalized dataset: {len(df)} rows, {len(df.columns)} columns")
        
        return df
    
    def handle_stock_splits(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """
        Adjust strikes and prices for historical stock splits
        
        Common splits:
        - AAPL: 4:1 split on 2020-08-31
        - TSLA: 5:1 split on 2020-08-31, 3:1 split on 2022-08-25
        - NVDA: 4:1 split on 2021-07-20, 10:1 split on 2024-06-10
        """
        # Known stock splits (date: split_ratio)
        STOCK_SPLITS = {
            'AAPL': [('2020-08-31', 4.0), ('2014-06-09', 7.0)],
            'TSLA': [('2020-08-31', 5.0), ('2022-08-25', 3.0)],
            'NVDA': [('2021-07-20', 4.0), ('2024-06-10', 10.0)],
            'GOOGL': [('2022-07-18', 20.0)],
            'AMZN': [('2022-06-06', 20.0)],
        }
        
        if ticker not in STOCK_SPLITS:
            return df
        
        if 'date' not in df.columns:
            return df
        
        df = df.copy()
        splits = STOCK_SPLITS[ticker]
        
        for split_date_str, ratio in splits:
            split_date = pd.to_datetime(split_date_str)
            
            # Adjust prices/strikes for data BEFORE the split
            # (data after split is already correct)
            mask = df['date'] < split_date
            
            if 'strike' in df.columns:
                df.loc[mask, 'strike'] = df.loc[mask, 'strike'] / ratio
            
            if 'close' in df.columns:
                df.loc[mask, 'close'] = df.loc[mask, 'close'] / ratio
            
            if 'open' in df.columns:
                df.loc[mask, 'open'] = df.loc[mask, 'open'] / ratio
            
            if 'high' in df.columns:
                df.loc[mask, 'high'] = df.loc[mask, 'high'] / ratio
            
            if 'low' in df.columns:
                df.loc[mask, 'low'] = df.loc[mask, 'low'] / ratio
            
            if 'bid' in df.columns:
                df.loc[mask, 'bid'] = df.loc[mask, 'bid'] / ratio
            
            if 'ask' in df.columns:
                df.loc[mask, 'ask'] = df.loc[mask, 'ask'] / ratio
            
            logger.info(f"Applied {ratio}:1 split adjustment for {ticker} on {split_date_str}")
        
        return df
    
    def calculate_implied_volatility(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """
        Calculate implied volatility using Black-Scholes
        
        Falls back to approximations if py_vollib not available
        """
        if 'implied_volatility' in df.columns:
            # Already have IV, just validate it
            df['implied_volatility'] = df['implied_volatility'].clip(
                QUALITY_THRESHOLDS['iv_min'],
                QUALITY_THRESHOLDS['iv_max']
            )
            return df
        
        df = df.copy()
        
        if VOLLIB_AVAILABLE:
            # Use Black-Scholes IV calculation
            df['implied_volatility'] = self._calculate_iv_blackscholes(df)
        else:
            # Use approximation based on option price and moneyness
            df['implied_volatility'] = self._approximate_iv(df)
        
        # Fill missing IVs with median
        if df['implied_volatility'].isnull().any():
            median_iv = df['implied_volatility'].median()
            df['implied_volatility'].fillna(median_iv, inplace=True)
            logger.warning(f"Filled {df['implied_volatility'].isnull().sum()} missing IVs with median: {median_iv:.4f}")
        
        # Clip to realistic range
        df['implied_volatility'] = df['implied_volatility'].clip(
            QUALITY_THRESHOLDS['iv_min'],
            QUALITY_THRESHOLDS['iv_max']
        )
        
        return df
    
    def _calculate_iv_blackscholes(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate IV using Black-Scholes and py_vollib
        
        Requires: option_price, price (underlying), strike, time_to_expiry, option_type
        """
        ivs = []
        
        required_cols = ['option_price', 'price', 'strike', 'time_to_expiry']
        if not all(col in df.columns for col in required_cols):
            logger.warning("Missing columns for IV calculation, using approximation")
            return self._approximate_iv(df)
        
        risk_free_rate = 0.03  # Approximate risk-free rate (3%)
        
        for idx, row in df.iterrows():
            try:
                S = row['price']  # Underlying price
                K = row['strike']  # Strike price
                t = row['time_to_expiry'] / 365.0  # Time in years
                option_price = row['option_price']
                flag = 'c' if row.get('option_type', 'C') == 'C' else 'p'
                
                if t <= 0 or S <= 0 or K <= 0 or option_price <= 0:
                    ivs.append(np.nan)
                    continue
                
                # Calculate IV using Newton-Raphson
                iv = iv_calc(
                    price=option_price,
                    S=S,
                    K=K,
                    t=t,
                    r=risk_free_rate,
                    flag=flag
                )
                
                ivs.append(iv)
                
            except Exception as e:
                # IV calculation failed (e.g., option too far OTM/ITM)
                ivs.append(np.nan)
        
        return pd.Series(ivs, index=df.index)
    
    def _approximate_iv(self, df: pd.DataFrame) -> pd.Series:
        """
        Approximate IV using option price and moneyness
        
        Simple approximation: IV ≈ option_price / (underlying_price * sqrt(time_to_expiry/365))
        """
        if 'option_price' not in df.columns or 'price' not in df.columns:
            # Return default IV
            return pd.Series(0.30, index=df.index)  # 30% default
        
        df_copy = df.copy()
        
        # Ensure positive values
        df_copy['option_price'] = df_copy['option_price'].clip(lower=0.01)
        df_copy['price'] = df_copy['price'].clip(lower=1.0)
        df_copy['time_to_expiry'] = df_copy.get('time_to_expiry', 30).clip(lower=1)
        
        # Simple IV approximation
        iv_approx = (
            df_copy['option_price'] / 
            (df_copy['price'] * np.sqrt(df_copy['time_to_expiry'] / 365.0))
        )
        
        # Adjust by moneyness factor
        if 'strike_distance' in df_copy.columns:
            moneyness = df_copy['strike_distance'] / df_copy['price']
            # OTM options tend to have higher IV (smile effect)
            iv_adjust = 1.0 + abs(moneyness) * 0.5
            iv_approx = iv_approx * iv_adjust
        
        # Clip to reasonable range
        iv_approx = iv_approx.clip(0.05, 3.0)
        
        return iv_approx
    
    def _merge_underlying_prices(self, df: pd.DataFrame, underlying_prices: pd.DataFrame) -> pd.DataFrame:
        """Merge underlying OHLCV data into options dataframe"""
        if underlying_prices.empty:
            logger.warning("Empty underlying prices, cannot merge")
            return df
        
        underlying = underlying_prices.copy()
        
        # Ensure date column
        if 'Date' in underlying.columns:
            underlying.rename(columns={'Date': 'date'}, inplace=True)
        
        underlying['date'] = pd.to_datetime(underlying['date'])
        
        # Rename columns to match schema
        rename_map = {
            'Close': 'price',
            'Volume': 'underlying_volume',
            'Open': 'underlying_open',
            'High': 'underlying_high',
            'Low': 'underlying_low',
        }
        
        underlying.rename(columns=rename_map, inplace=True)
        
        # Merge on date
        df = df.merge(
            underlying[['date', 'price', 'underlying_volume']],
            on='date',
            how='left'
        )
        
        return df
    
    def _normalize_volume(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize volume using log1p and rolling median
        
        Handles volume scale differences across years
        """
        if 'volume' not in df.columns:
            return df
        
        df = df.copy()
        
        # Log transform (handles zeros)
        df['volume_log'] = np.log1p(df['volume'])
        
        # Rolling median normalization (252-day window ≈ 1 year)
        df['volume_median_252d'] = df['volume'].rolling(window=252, min_periods=50).median()
        df['volume_normalized'] = df['volume'] / df['volume_median_252d'].clip(lower=1)
        
        # Keep original volume for reference, use normalized for features
        df.rename(columns={'volume': 'volume_raw'}, inplace=True)
        df.rename(columns={'volume_log': 'volume'}, inplace=True)
        
        return df
    
    def _add_vix_level(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fetch and merge VIX levels"""
        if 'vix_level' in df.columns:
            return df
        
        if df.empty or 'date' not in df.columns:
            df['vix_level'] = 20.0  # Default VIX
            return df
        
        # Fetch VIX data if not cached
        if self.vix_cache is None:
            logger.info("Fetching VIX data from Yahoo Finance...")
            try:
                start_date = df['date'].min() - timedelta(days=10)
                end_date = df['date'].max() + timedelta(days=1)
                
                vix = yf.download('^VIX', start=start_date, end=end_date, progress=False)
                
                if not vix.empty:
                    vix = vix.reset_index()
                    vix.columns = ['date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
                    vix['date'] = pd.to_datetime(vix['date'])
                    vix['vix_level'] = vix['Close']
                    self.vix_cache = vix[['date', 'vix_level']]
                else:
                    logger.warning("Failed to fetch VIX data, using default")
                    df['vix_level'] = 20.0
                    return df
                    
            except Exception as e:
                logger.error(f"Error fetching VIX: {e}")
                df['vix_level'] = 20.0
                return df
        
        # Merge VIX data
        df = df.merge(self.vix_cache, on='date', how='left')
        
        # Forward fill missing VIX values
        df['vix_level'].fillna(method='ffill', inplace=True)
        df['vix_level'].fillna(20.0, inplace=True)  # Default for any remaining
        
        return df
    
    def _calculate_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate returns as proxy for spy_return_5min
        
        Uses daily returns since dataset is EOD (no intraday data)
        """
        if 'price' not in df.columns:
            df['spy_return_5min'] = 0.0
            df['spy_return_1d'] = 0.0
            return df
        
        df = df.copy()
        
        # Sort by date to ensure proper return calculation
        df = df.sort_values('date')
        
        # Calculate daily return on underlying
        df['daily_return'] = df['price'].pct_change()
        
        # Use daily return as proxy for short-term return
        df['spy_return_5min'] = df['daily_return'].fillna(0.0)
        df['spy_return_1d'] = df['spy_return_5min']
        
        return df
    
    def _calculate_rsi(self, df: pd.DataFrame, underlying_prices: pd.DataFrame) -> pd.DataFrame:
        """Calculate RSI (Relative Strength Index) on underlying"""
        if 'rsi' in df.columns:
            return df
        
        if underlying_prices.empty:
            df['rsi'] = 50.0  # Neutral RSI
            return df
        
        # Calculate RSI from underlying price data
        underlying = underlying_prices.copy()
        
        if 'Date' in underlying.columns:
            underlying.rename(columns={'Date': 'date'}, inplace=True)
        
        if 'Close' in underlying.columns:
            underlying['date'] = pd.to_datetime(underlying['date'])
            
            # Calculate price changes
            delta = underlying['Close'].diff()
            
            # Separate gains and losses
            gain = delta.where(delta > 0, 0.0)
            loss = -delta.where(delta < 0, 0.0)
            
            # Calculate rolling average gain/loss (14-day window)
            window = 14
            avg_gain = gain.rolling(window=window, min_periods=1).mean()
            avg_loss = loss.rolling(window=window, min_periods=1).mean()
            
            # RSI calculation
            rs = avg_gain / avg_loss.clip(lower=0.0001)  # Avoid division by zero
            rsi = 100 - (100 / (1 + rs))
            
            underlying['rsi'] = rsi
            
            # Merge RSI back to options dataframe
            df = df.merge(underlying[['date', 'rsi']], on='date', how='left')
            
            # Fill missing RSI
            df['rsi'].fillna(method='ffill', inplace=True)
            df['rsi'].fillna(50.0, inplace=True)
        else:
            df['rsi'] = 50.0
        
        return df
    
    def _create_timestamp(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create timestamp column from date"""
        if 'timestamp' in df.columns:
            return df
        
        if 'date' not in df.columns:
            df['timestamp'] = datetime.now()
            return df
        
        df = df.copy()
        
        # Convert date to timestamp (add 16:00:00 for market close)
        df['timestamp'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d 16:00:00')
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        return df
    
    def _filter_to_schema(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Filter and order columns to match Training.py schema"""
        df = df.copy()
        
        # Add ticker column (useful for multi-ticker datasets)
        df['ticker'] = ticker
        
        # Ensure all required columns exist
        for col in REQUIRED_COLUMNS:
            if col not in df.columns:
                logger.warning(f"Missing required column: {col}, filling with defaults")
                
                # Fill with reasonable defaults
                if col == 'implied_volatility':
                    df[col] = 0.30
                elif col == 'vix_level':
                    df[col] = 20.0
                elif col == 'rsi':
                    df[col] = 50.0
                elif col in ['spy_return_5min', 'spy_return_1d', 'strike_distance']:
                    df[col] = 0.0
                elif col == 'time_to_expiry':
                    df[col] = 30
                elif col == 'timestamp':
                    df[col] = datetime.now()
                else:
                    df[col] = 0.0
        
        # Select required columns + ticker
        output_cols = REQUIRED_COLUMNS + ['ticker']
        
        # Add optional columns if they exist
        for col in OPTIONAL_COLUMNS:
            if col in df.columns and col not in output_cols:
                output_cols.append(col)
        
        # Filter to available columns
        output_cols = [col for col in output_cols if col in df.columns]
        
        return df[output_cols]
    
    def _apply_quality_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply data quality filters"""
        if df.empty:
            return df
        
        initial_rows = len(df)
        
        # Filter 1: Remove options too close to expiry
        if 'time_to_expiry' in df.columns:
            df = df[df['time_to_expiry'] >= QUALITY_THRESHOLDS['min_dte']]
            df = df[df['time_to_expiry'] <= QUALITY_THRESHOLDS['max_dte']]
        
        # Filter 2: Remove invalid prices
        if 'option_price' in df.columns:
            df = df[df['option_price'] > 0]
        
        if 'price' in df.columns:
            df = df[df['price'] > 0]
        
        # Filter 3: Remove extreme IVs
        if 'implied_volatility' in df.columns:
            df = df[
                (df['implied_volatility'] >= QUALITY_THRESHOLDS['iv_min']) &
                (df['implied_volatility'] <= QUALITY_THRESHOLDS['iv_max'])
            ]
        
        # Filter 4: Remove rows with too many missing values
        max_missing = int(len(REQUIRED_COLUMNS) * QUALITY_THRESHOLDS['max_missing_columns_pct'])
        df = df.dropna(subset=REQUIRED_COLUMNS, thresh=len(REQUIRED_COLUMNS) - max_missing)
        
        removed_rows = initial_rows - len(df)
        if removed_rows > 0:
            logger.info(f"Removed {removed_rows} rows ({removed_rows/initial_rows:.1%}) due to quality filters")
        
        return df


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def validate_normalized_data(df: pd.DataFrame) -> Dict[str, any]:
    """
    Validate normalized dataset against Training.py requirements
    
    Returns:
        Dictionary with validation results
    """
    results = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'stats': {}
    }
    
    # Check required columns
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        results['valid'] = False
        results['errors'].append(f"Missing required columns: {missing_cols}")
    
    # Check data completeness
    if not df.empty:
        for col in REQUIRED_COLUMNS:
            if col in df.columns:
                missing_pct = df[col].isnull().sum() / len(df)
                if missing_pct > QUALITY_THRESHOLDS['max_missing_columns_pct']:
                    results['warnings'].append(
                        f"Column {col} has {missing_pct:.1%} missing values"
                    )
        
        # Stats
        results['stats'] = {
            'total_rows': len(df),
            'date_range': (df['timestamp'].min(), df['timestamp'].max()) if 'timestamp' in df.columns else None,
            'tickers': df['ticker'].unique().tolist() if 'ticker' in df.columns else [],
            'avg_dte': df['time_to_expiry'].mean() if 'time_to_expiry' in df.columns else None,
            'avg_iv': df['implied_volatility'].mean() if 'implied_volatility' in df.columns else None,
        }
    
    return results


if __name__ == '__main__':
    # Test normalizer
    print("Testing MultiYearNormalizer...")
    
    normalizer = MultiYearNormalizer()
    
    # Create sample raw data
    sample_data = pd.DataFrame({
        'date': pd.date_range('2020-01-01', periods=100),
        'close': np.random.uniform(5, 15, 100),
        'volume': np.random.randint(100, 10000, 100),
        'bid': np.random.uniform(4, 14, 100),
        'ask': np.random.uniform(6, 16, 100),
        'strike': 150.0,
        'expiration_date': '2020-03-20',
        'option_type': 'C',
    })
    
    # Create sample underlying prices
    underlying = pd.DataFrame({
        'date': pd.date_range('2020-01-01', periods=100),
        'Close': np.random.uniform(145, 155, 100),
        'Volume': np.random.randint(1000000, 5000000, 100),
    })
    
    # Normalize
    normalized = normalizer.normalize_dataset(sample_data, 'TEST', underlying)
    
    print(f"\nNormalized {len(normalized)} rows")
    print(f"Columns: {normalized.columns.tolist()}")
    print(f"\nSample data:")
    print(normalized.head())
    
    # Validate
    validation = validate_normalized_data(normalized)
    print(f"\nValidation: {'✅ PASSED' if validation['valid'] else '❌ FAILED'}")
    if validation['errors']:
        print(f"Errors: {validation['errors']}")
    if validation['warnings']:
        print(f"Warnings: {validation['warnings']}")
    print(f"Stats: {validation['stats']}")
