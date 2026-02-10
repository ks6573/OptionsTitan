"""
Historical Data Fetcher (DEPRECATED - LEGACY CODE)

⚠️ DEPRECATED: This module was built for paid API services and is no longer maintained.

✅ USE INSTEAD: src/data_collection/remote_query.py for FREE data collection

This legacy code orchestrated multi-year, multi-ticker options data collection using:
- Paid API service for options data
- yfinance for underlying prices and VIX
- Data normalizer for schema transformation
- Parquet storage with partitioning

Keeping for reference only. Does not work without external API client.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
import os
import json
from pathlib import Path
import yfinance as yf
import time

from .thetadata_client import ThetaDataClient, format_date_thetadata, find_closest_expiration, find_atm_strike
from .data_normalizer import MultiYearNormalizer, validate_normalized_data
from .config import (
    ALL_TICKERS, TICKER_UNIVERSE, SAMPLING_CONFIG, DATA_RANGE,
    STORAGE_CONFIG, QUALITY_THRESHOLDS, get_ticker_category
)

logger = logging.getLogger(__name__)


class HistoricalDataFetcher:
    """
    Fetch and normalize multi-year historical options data
    
    Workflow:
    1. Fetch underlying price history (yfinance)
    2. For each trading day:
       - Find target expiries (DTE buckets)
       - Calculate ATM and offset strikes
       - Fetch 20 contracts (2 exp × 5 strikes × 2 types)
    3. Fetch supplementary data (VIX)
    4. Normalize to Training.py schema
    5. Save as partitioned Parquet + CSV
    """
    
    def __init__(self, theta_client: ThetaDataClient = None):
        """
        Initialize fetcher
        
        Args:
            theta_client: ThetaDataClient instance (creates new if None)
        """
        self.theta_client = theta_client or ThetaDataClient()
        self.normalizer = MultiYearNormalizer()
        
        # Create storage directories
        self._create_directories()
        
        # Load fetch progress (for resume capability)
        self.progress = self._load_progress()
    
    def _create_directories(self):
        """Create storage directory structure"""
        for dir_path in [STORAGE_CONFIG['raw_parquet_dir'],
                         STORAGE_CONFIG['processed_csv_dir'],
                         STORAGE_CONFIG['metadata_dir']]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
        
        logger.info("Storage directories ready")
    
    def _load_progress(self) -> Dict:
        """Load fetch progress from metadata"""
        progress_file = Path(STORAGE_CONFIG['metadata_dir']) / 'fetch_progress.json'
        
        if progress_file.exists():
            with open(progress_file, 'r') as f:
                return json.load(f)
        
        return {'completed_tickers': [], 'failed_tickers': []}
    
    def _save_progress(self):
        """Save fetch progress to metadata"""
        progress_file = Path(STORAGE_CONFIG['metadata_dir']) / 'fetch_progress.json'
        
        with open(progress_file, 'w') as f:
            json.dump(self.progress, f, indent=2)
    
    def fetch_single_ticker(self,
                           ticker: str,
                           start_date: str,
                           end_date: str,
                           save_parquet: bool = True,
                           save_csv: bool = True) -> pd.DataFrame:
        """
        Fetch all sampled contracts for one ticker over date range
        
        Args:
            ticker: Stock symbol (e.g., 'AAPL')
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            save_parquet: Save raw data to Parquet
            save_csv: Save normalized data to CSV
        
        Returns:
            Normalized DataFrame with Training.py schema
        """
        logger.info(f"=" * 70)
        logger.info(f"Fetching {ticker}: {start_date} to {end_date}")
        logger.info(f"=" * 70)
        
        # Step 1: Fetch underlying price history
        logger.info(f"Step 1/6: Fetching underlying prices for {ticker}...")
        underlying_prices = self._fetch_underlying_prices(ticker, start_date, end_date)
        
        if underlying_prices.empty:
            logger.error(f"Failed to fetch underlying prices for {ticker}")
            return pd.DataFrame()
        
        logger.info(f"  ✓ Fetched {len(underlying_prices)} days of underlying data")
        
        # Step 2: Get available expirations from ThetaData
        logger.info(f"Step 2/6: Fetching available expirations...")
        try:
            all_expirations = self.theta_client.list_expirations(ticker)
            logger.info(f"  ✓ Found {len(all_expirations)} total expirations")
        except Exception as e:
            logger.error(f"Failed to fetch expirations: {e}")
            return pd.DataFrame()
        
        # Step 3: Sample contracts across trading days
        logger.info(f"Step 3/6: Sampling option contracts...")
        contracts_to_fetch = self._sample_contracts(
            ticker, underlying_prices, all_expirations, start_date, end_date
        )
        
        if not contracts_to_fetch:
            logger.error("No contracts to fetch")
            return pd.DataFrame()
        
        logger.info(f"  ✓ Selected {len(contracts_to_fetch)} contracts to fetch")
        
        # Step 4: Fetch option data from ThetaData
        logger.info(f"Step 4/6: Fetching option data from ThetaData...")
        raw_options_data = self._fetch_option_contracts(
            ticker, contracts_to_fetch, start_date, end_date
        )
        
        if raw_options_data.empty:
            logger.warning(f"No option data fetched for {ticker}")
            return pd.DataFrame()
        
        logger.info(f"  ✓ Fetched {len(raw_options_data)} rows of option data")
        
        # Step 5: Normalize to Training.py schema
        logger.info(f"Step 5/6: Normalizing data to Training.py schema...")
        normalized_data = self.normalizer.normalize_dataset(
            raw_options_data, ticker, underlying_prices
        )
        
        logger.info(f"  ✓ Normalized to {len(normalized_data)} rows")
        
        # Step 6: Save data
        if save_parquet and not raw_options_data.empty:
            logger.info(f"Step 6/6: Saving data...")
            self._save_to_parquet(raw_options_data, ticker)
            logger.info(f"  ✓ Saved raw Parquet")
        
        if save_csv and not normalized_data.empty:
            self._save_to_csv(normalized_data, ticker, start_date, end_date)
            logger.info(f"  ✓ Saved normalized CSV")
        
        # Validation
        validation = validate_normalized_data(normalized_data)
        logger.info(f"\nValidation: {'✅ PASSED' if validation['valid'] else '⚠️ WARNINGS'}")
        if validation['warnings']:
            for warning in validation['warnings']:
                logger.warning(f"  {warning}")
        
        # Mark as completed
        if ticker not in self.progress['completed_tickers']:
            self.progress['completed_tickers'].append(ticker)
            self._save_progress()
        
        logger.info(f"=" * 70)
        logger.info(f"✅ Completed {ticker}")
        logger.info(f"=" * 70)
        
        return normalized_data
    
    def fetch_all_tickers(self,
                         start_date: str = None,
                         end_date: str = None,
                         resume: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Fetch all 45 tickers from universe
        
        Args:
            start_date: Start date (default: from config)
            end_date: End date (default: from config)
            resume: Skip already completed tickers
        
        Returns:
            Dictionary of {ticker: DataFrame}
        """
        start_date = start_date or DATA_RANGE['start_date']
        end_date = end_date or DATA_RANGE['end_date']
        
        logger.info(f"\n{'='*70}")
        logger.info(f"FETCHING ALL TICKERS: {start_date} to {end_date}")
        logger.info(f"{'='*70}")
        logger.info(f"Total tickers: {len(ALL_TICKERS)}")
        
        if resume:
            completed = self.progress['completed_tickers']
            logger.info(f"Already completed: {len(completed)} tickers")
            remaining_tickers = [t for t in ALL_TICKERS if t not in completed]
        else:
            remaining_tickers = ALL_TICKERS
        
        logger.info(f"To fetch: {len(remaining_tickers)} tickers")
        logger.info(f"{'='*70}\n")
        
        results = {}
        failed_tickers = []
        
        for i, ticker in enumerate(remaining_tickers, 1):
            logger.info(f"\n[{i}/{len(remaining_tickers)}] Processing {ticker}...")
            
            try:
                df = self.fetch_single_ticker(ticker, start_date, end_date)
                results[ticker] = df
                
                # Log progress
                logger.info(f"✅ {ticker} completed ({len(df)} rows)")
                
            except Exception as e:
                logger.error(f"❌ {ticker} failed: {e}")
                failed_tickers.append(ticker)
                self.progress['failed_tickers'].append(ticker)
                self._save_progress()
                
                # Continue with next ticker
                continue
        
        # Final summary
        logger.info(f"\n{'='*70}")
        logger.info(f"FETCH SUMMARY")
        logger.info(f"{'='*70}")
        logger.info(f"Successful: {len(results)}/{len(remaining_tickers)}")
        logger.info(f"Failed: {len(failed_tickers)}")
        if failed_tickers:
            logger.info(f"Failed tickers: {', '.join(failed_tickers)}")
        logger.info(f"{'='*70}\n")
        
        return results
    
    def _fetch_underlying_prices(self,
                                ticker: str,
                                start_date: str,
                                end_date: str) -> pd.DataFrame:
        """Fetch underlying OHLCV from yfinance"""
        try:
            # Add buffer for technical indicators
            start_dt = datetime.strptime(start_date, '%Y-%m-%d') - timedelta(days=60)
            
            ticker_obj = yf.Ticker(ticker)
            df = ticker_obj.history(
                start=start_dt.strftime('%Y-%m-%d'),
                end=end_date,
                interval='1d'
            )
            
            if df.empty:
                logger.warning(f"No underlying data from yfinance for {ticker}")
                return pd.DataFrame()
            
            # Reset index to get Date as column
            df = df.reset_index()
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching underlying prices: {e}")
            return pd.DataFrame()
    
    def _sample_contracts(self,
                         ticker: str,
                         underlying_prices: pd.DataFrame,
                         all_expirations: List[str],
                         start_date: str,
                         end_date: str) -> List[Dict]:
        """
        Sample 20 contracts per trading day
        
        Strategy:
        - 2 expiries: one in (25-35 DTE), one in (45-75 DTE)
        - 5 strikes per expiry: ATM ± 5%, ±2.5%, 0%
        - 2 types: Call and Put
        = 2 × 5 × 2 = 20 contracts/day
        
        Returns:
            List of contract specifications: [
                {'date': date, 'exp': exp, 'strike': strike, 'right': right},
                ...
            ]
        """
        contracts = []
        
        # Filter underlying prices to date range
        underlying_prices['Date'] = pd.to_datetime(underlying_prices['Date'])
        mask = (
            (underlying_prices['Date'] >= start_date) &
            (underlying_prices['Date'] <= end_date)
        )
        trading_days = underlying_prices[mask]
        
        logger.info(f"  Sampling from {len(trading_days)} trading days")
        
        # Sample every Nth day to reduce data volume (optional)
        # For POC: sample all days; for production: maybe every 2-3 days
        sample_frequency = 1  # Sample every day
        trading_days = trading_days.iloc[::sample_frequency]
        
        for idx, row in trading_days.iterrows():
            date = row['Date']
            underlying_price = row['Close']
            
            # Find expirations in target DTE buckets
            for dte_min, dte_max in SAMPLING_CONFIG['expiry_buckets']:
                # Find expiration closest to target DTE midpoint
                target_dte = (dte_min + dte_max) / 2
                
                exp = find_closest_expiration(
                    all_expirations,
                    target_dte,
                    date
                )
                
                if exp is None:
                    continue
                
                # Calculate strikes around ATM
                for moneyness_pct in SAMPLING_CONFIG['moneyness_pct']:
                    strike_offset = underlying_price * (moneyness_pct / 100.0)
                    strike = underlying_price + strike_offset
                    
                    # Round to nearest $1 or $5 depending on price level
                    if underlying_price < 50:
                        strike = round(strike, 1)  # $0.10 increments -> round to nearest
                    elif underlying_price < 200:
                        strike = round(strike / 0.5) * 0.5  # $0.50 increments
                    else:
                        strike = round(strike)  # $1 increments
                    
                    # Add both call and put
                    for right in SAMPLING_CONFIG['option_types']:
                        contracts.append({
                            'date': date,
                            'exp': exp,
                            'strike': strike,
                            'right': right,
                            'underlying_price': underlying_price
                        })
        
        logger.info(f"  Generated {len(contracts)} contract specifications")
        
        return contracts
    
    def _fetch_option_contracts(self,
                               ticker: str,
                               contracts: List[Dict],
                               start_date: str,
                               end_date: str) -> pd.DataFrame:
        """
        Fetch option data for sampled contracts
        
        Returns:
            Raw option data with ThetaData fields + contract metadata
        """
        all_data = []
        
        # Group contracts by expiration for efficient fetching
        contracts_by_exp = {}
        for contract in contracts:
            exp = contract['exp']
            if exp not in contracts_by_exp:
                contracts_by_exp[exp] = []
            contracts_by_exp[exp].append(contract)
        
        total_contracts = len(contracts)
        fetched_count = 0
        
        logger.info(f"  Fetching {total_contracts} contracts across {len(contracts_by_exp)} expirations...")
        
        for exp_idx, (exp, exp_contracts) in enumerate(contracts_by_exp.items(), 1):
            logger.info(f"    Expiration {exp_idx}/{len(contracts_by_exp)}: {exp} ({len(exp_contracts)} contracts)")
            
            for contract in exp_contracts:
                try:
                    # Fetch this contract's historical data
                    df = self.theta_client.fetch_option_eod(
                        root=ticker,
                        exp=exp,
                        strike=contract['strike'],
                        right=contract['right'],
                        start_date=format_date_thetadata(start_date),
                        end_date=format_date_thetadata(end_date)
                    )
                    
                    if not df.empty:
                        # Add contract metadata
                        df['ticker'] = ticker
                        df['strike'] = contract['strike']
                        df['expiration_date'] = datetime.strptime(exp, '%Y%m%d')
                        df['option_type'] = contract['right']
                        
                        all_data.append(df)
                        fetched_count += 1
                    
                except Exception as e:
                    logger.warning(f"    Failed to fetch {ticker} {exp} {contract['strike']} {contract['right']}: {e}")
                    continue
            
            # Log progress periodically
            if exp_idx % 10 == 0:
                logger.info(f"    Progress: {fetched_count}/{total_contracts} contracts fetched")
        
        if not all_data:
            logger.warning("  No option data fetched")
            return pd.DataFrame()
        
        # Combine all data
        combined_df = pd.concat(all_data, ignore_index=True)
        
        logger.info(f"  ✓ Successfully fetched {fetched_count}/{total_contracts} contracts")
        logger.info(f"  ✓ Total rows: {len(combined_df)}")
        
        return combined_df
    
    def _save_to_parquet(self, df: pd.DataFrame, ticker: str):
        """Save raw data to partitioned Parquet"""
        if df.empty:
            return
        
        # Add partitioning columns
        df = df.copy()
        if 'date' in df.columns:
            df['year'] = pd.to_datetime(df['date']).dt.year
        elif 'timestamp' in df.columns:
            df['year'] = pd.to_datetime(df['timestamp']).dt.year
        else:
            df['year'] = datetime.now().year
        
        df['ticker'] = ticker
        
        # Save partitioned by ticker and year
        base_dir = Path(STORAGE_CONFIG['raw_parquet_dir'])
        
        for year in df['year'].unique():
            year_df = df[df['year'] == year]
            
            partition_dir = base_dir / f"ticker={ticker}" / f"year={year}"
            partition_dir.mkdir(parents=True, exist_ok=True)
            
            filepath = partition_dir / "contracts.parquet"
            
            year_df.to_parquet(
                filepath,
                compression=STORAGE_CONFIG['parquet_compression'],
                index=False
            )
            
            logger.info(f"  Saved {len(year_df)} rows to {filepath}")
    
    def _save_to_csv(self, df: pd.DataFrame, ticker: str, start_date: str, end_date: str):
        """Save normalized data to CSV"""
        if df.empty:
            return
        
        csv_dir = Path(STORAGE_CONFIG['processed_csv_dir'])
        csv_dir.mkdir(parents=True, exist_ok=True)
        
        # Create filename
        start_year = start_date[:4]
        end_year = end_date[:4]
        filename = f"{ticker.lower()}_{start_year}_{end_year}_options.csv"
        filepath = csv_dir / filename
        
        # Save CSV
        df.to_csv(
            filepath,
            index=False,
            float_format=STORAGE_CONFIG['csv_float_format'],
            date_format=STORAGE_CONFIG['csv_date_format']
        )
        
        logger.info(f"  Saved CSV: {filepath} ({len(df)} rows)")
    
    def get_stats(self) -> Dict:
        """Get fetcher statistics"""
        return {
            'completed_tickers': len(self.progress['completed_tickers']),
            'failed_tickers': len(self.progress['failed_tickers']),
            'theta_client_stats': self.theta_client.get_stats()
        }


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def main():
    """Command-line interface for data fetching"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Fetch historical options data from ThetaData'
    )
    
    parser.add_argument(
        '--ticker',
        type=str,
        help='Single ticker to fetch (e.g., AAPL). If omitted, fetches all tickers.'
    )
    
    parser.add_argument(
        '--start',
        type=str,
        default=DATA_RANGE['start_date'],
        help=f'Start date (YYYY-MM-DD). Default: {DATA_RANGE["start_date"]}'
    )
    
    parser.add_argument(
        '--end',
        type=str,
        default=DATA_RANGE['end_date'],
        help=f'End date (YYYY-MM-DD). Default: {DATA_RANGE["end_date"]}'
    )
    
    parser.add_argument(
        '--no-resume',
        action='store_true',
        help='Disable resume (re-fetch completed tickers)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create fetcher
    logger.info("Initializing ThetaData client...")
    try:
        fetcher = HistoricalDataFetcher()
        logger.info("✅ Connected to ThetaData Terminal")
    except Exception as e:
        logger.error(f"❌ Failed to connect: {e}")
        logger.error("\nMake sure:")
        logger.error("1. ThetaData Terminal is installed")
        logger.error("2. Terminal is running")
        logger.error("3. You're logged in")
        return
    
    # Fetch data
    try:
        if args.ticker:
            # Single ticker
            logger.info(f"\nFetching single ticker: {args.ticker}")
            df = fetcher.fetch_single_ticker(
                args.ticker.upper(),
                args.start,
                args.end
            )
            logger.info(f"\n✅ Completed! Fetched {len(df)} rows")
        
        else:
            # All tickers
            logger.info(f"\nFetching all {len(ALL_TICKERS)} tickers")
            results = fetcher.fetch_all_tickers(
                args.start,
                args.end,
                resume=not args.no_resume
            )
            
            total_rows = sum(len(df) for df in results.values())
            logger.info(f"\n✅ Completed! Fetched {total_rows} total rows across {len(results)} tickers")
        
        # Print stats
        stats = fetcher.get_stats()
        logger.info(f"\nStatistics:")
        logger.info(f"  Completed tickers: {stats['completed_tickers']}")
        logger.info(f"  Failed tickers: {stats['failed_tickers']}")
        logger.info(f"  API requests: {stats['theta_client_stats']['total_requests']}")
        
    except KeyboardInterrupt:
        logger.info("\n\n⚠️  Interrupted by user. Progress saved.")
        logger.info("Run again with same arguments to resume.")
    
    except Exception as e:
        logger.error(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
