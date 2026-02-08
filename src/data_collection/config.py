"""
Data Collection Configuration

Defines the ticker universe, sampling parameters, and date ranges for
multi-year historical options data collection.

Based on GPT-5 recommendations for diverse market coverage:
- 45 tickers across sectors and volatility regimes
- 20 contracts per trading day per ticker
- 2019-2024 historical range (5+ years including COVID)
"""

from typing import Dict, List, Tuple
from datetime import datetime

# ============================================================================
# TICKER UNIVERSE (45 tickers)
# ============================================================================

TICKER_UNIVERSE: Dict[str, List[str]] = {
    # Liquid ETFs with high options volume
    'ETFs': [
        'SPY',   # S&P 500 - Ultra liquid, benchmark
        'QQQ',   # Nasdaq 100 - Tech-heavy
        'IWM',   # Russell 2000 - Small caps
        'DIA',   # Dow Jones Industrial
        'XLF',   # Financial Sector
        'XLK',   # Technology Sector
        'XLE',   # Energy Sector
    ],
    
    # Mega-cap tech with high options activity
    'Mega_Tech': [
        'AAPL',   # Apple - Most liquid stock options
        'MSFT',   # Microsoft
        'NVDA',   # Nvidia - High volatility tech
        'AMZN',   # Amazon
        'GOOGL',  # Alphabet
        'META',   # Meta (formerly Facebook)
        'TSLA',   # Tesla - Extremely high volatility
        'AMD',    # Advanced Micro Devices
        'NFLX',   # Netflix
        'ORCL',   # Oracle
    ],
    
    # Major financials
    'Financials': [
        'JPM',    # JP Morgan
        'BAC',    # Bank of America
        'GS',     # Goldman Sachs
        'MS',     # Morgan Stanley
        'WFC',    # Wells Fargo
        'SCHW',   # Charles Schwab
    ],
    
    # Healthcare and pharma
    'Healthcare': [
        'UNH',    # United Health
        'JNJ',    # Johnson & Johnson
        'PFE',    # Pfizer
        'ABBV',   # AbbVie
        'LLY',    # Eli Lilly
    ],
    
    # Energy sector
    'Energy': [
        'XOM',    # Exxon Mobil
        'CVX',    # Chevron
        'COP',    # ConocoPhillips
        'SLB',    # Schlumberger
    ],
    
    # Consumer staples and discretionary
    'Consumer': [
        'WMT',    # Walmart
        'COST',   # Costco
        'TGT',    # Target
        'DIS',    # Disney
        'NKE',    # Nike
    ],
    
    # High-volatility / meme stocks
    'High_Vol': [
        'GME',    # GameStop - Extreme volatility
        'AMC',    # AMC Entertainment
        'PLTR',   # Palantir
        'COIN',   # Coinbase - Crypto proxy
        'ROKU',   # Roku
        'SNAP',   # Snap Inc
        'DKNG',   # DraftKings
        'MARA',   # Marathon Digital - Bitcoin mining
    ],
    
    # Low-volatility / stable dividend stocks
    'Low_Vol': [
        'KO',     # Coca-Cola
        'PG',     # Procter & Gamble
        'PEP',    # PepsiCo
        'MCD',    # McDonald's
        'V',      # Visa
    ],
}

# Flatten ticker list for convenience
ALL_TICKERS: List[str] = [
    ticker 
    for category_tickers in TICKER_UNIVERSE.values() 
    for ticker in category_tickers
]

# Ticker metadata (volume estimates, IV bands, sector info)
TICKER_METADATA: Dict[str, Dict] = {
    # Will be populated dynamically or from external source
    # Format: {ticker: {'sector': str, 'avg_volume': int, 'iv_band': str}}
}

# ============================================================================
# SAMPLING CONFIGURATION
# ============================================================================

SAMPLING_CONFIG: Dict = {
    # Expiry buckets: DTE (Days To Expiry) ranges
    # Target 2 expiry buckets per trading day
    'expiry_buckets': [
        (25, 35),   # ~1 month out (standard monthly)
        (45, 75),   # ~2 months out (quarterly-ish)
    ],
    
    # Moneyness: % offset from ATM strike
    # Creates a bracket around ATM to capture ITM/OTM behavior
    'moneyness_pct': [-5.0, -2.5, 0.0, 2.5, 5.0],
    
    # Option types to fetch
    'option_types': ['C', 'P'],  # Call and Put
    
    # Liquidity filters (minimum thresholds)
    'min_volume': 0,              # No volume filter (include all for completeness)
    'min_open_interest': 100,     # Require some OI for realistic contracts
    
    # Contracts per day per ticker
    # 2 expiries × 5 strikes × 2 types = 20 contracts/day
    'contracts_per_day': 20,
}

# ============================================================================
# DATE RANGE CONFIGURATION
# ============================================================================

DATA_RANGE: Dict[str, str] = {
    # Historical data fetch range
    'start_date': '2019-01-01',  # Pre-COVID baseline
    'end_date': '2024-12-31',    # ~5 years of data
    
    # Market regime periods (for analysis and balancing)
    'regimes': {
        'pre_covid': ('2019-01-01', '2020-02-28'),
        'covid_crash': ('2020-03-01', '2020-04-30'),
        'covid_recovery': ('2020-05-01', '2021-12-31'),
        'rate_hikes': ('2022-01-01', '2023-12-31'),
        'current': ('2024-01-01', '2024-12-31'),
    }
}

# ============================================================================
# THETADATA API CONFIGURATION
# ============================================================================

THETADATA_CONFIG: Dict = {
    # ThetaData Terminal REST API endpoint
    'base_url': 'http://127.0.0.1:25510',
    
    # API version
    'api_version': 'v2',
    
    # Rate limiting (requests per second)
    # Free tier: ~10/sec, Standard: ~100/sec
    'rate_limit_rps': 10,  # Conservative default
    
    # Retry configuration
    'max_retries': 3,
    'retry_delay': 1.0,  # seconds
    'backoff_factor': 2.0,  # Exponential backoff
    
    # Timeout configuration
    'request_timeout': 30,  # seconds per request
    
    # Batch size for bulk fetching
    'batch_size': 100,  # Contracts per batch
}

# ============================================================================
# STORAGE CONFIGURATION
# ============================================================================

STORAGE_CONFIG: Dict = {
    # Directory structure
    'raw_parquet_dir': 'data/raw_parquet',
    'processed_csv_dir': 'data/processed_csv',
    'metadata_dir': 'data/metadata',
    
    # Parquet partitioning scheme
    # Format: ticker=XXX/year=YYYY/
    'partition_cols': ['ticker', 'year'],
    
    # Compression
    'parquet_compression': 'snappy',  # Fast compression
    
    # CSV format (for Training.py compatibility)
    'csv_float_format': '%.6f',
    'csv_date_format': '%Y-%m-%d %H:%M:%S',
}

# ============================================================================
# SCHEMA DEFINITION (Training.py compatibility)
# ============================================================================

REQUIRED_COLUMNS: List[str] = [
    'price',              # Underlying price
    'option_price',       # Option premium (close or mid)
    'strike_distance',    # strike - underlying_price
    'time_to_expiry',     # Days to expiration (DTE)
    'volume',             # Option volume
    'implied_volatility', # IV (calculated or fetched)
    'vix_level',          # VIX index level
    'spy_return_5min',    # Short-term return proxy (use daily)
    'rsi',                # RSI indicator on underlying
    'timestamp',          # YYYY-MM-DD HH:MM:SS
]

# Additional columns to capture (not required by Training.py but useful)
OPTIONAL_COLUMNS: List[str] = [
    'ticker',             # Stock symbol
    'sector',             # Sector classification
    'contract_symbol',    # Full option symbol (OCC format)
    'expiration_date',    # Option expiration date
    'strike',             # Strike price
    'option_type',        # 'C' or 'P'
    'bid',                # Bid price
    'ask',                # Ask price
    'bid_size',           # Bid size
    'ask_size',           # Ask size
    'open_interest',      # Open interest
    'open',               # Option open price
    'high',               # Option high price
    'low',                # Option low price
    'underlying_volume',  # Underlying stock volume
    'date',               # Calendar date (YYYY-MM-DD)
]

# ============================================================================
# VALIDATION THRESHOLDS
# ============================================================================

QUALITY_THRESHOLDS: Dict = {
    # Data completeness
    'min_trading_days_pct': 0.90,  # Require 90% of trading days
    'max_missing_columns_pct': 0.05,  # Allow 5% missing per column
    
    # Outlier detection
    'price_change_max_pct': 0.30,  # Flag >30% daily price changes
    'volume_zscore_threshold': 5.0,  # Flag extreme volume spikes
    'iv_min': 0.01,  # Minimum realistic IV (1%)
    'iv_max': 5.00,  # Maximum realistic IV (500%)
    
    # Contract selection
    'min_dte': 7,    # Skip options <7 days (theta decay too high)
    'max_dte': 180,  # Skip options >180 days (low liquidity)
}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_tickers_by_category(category: str) -> List[str]:
    """Get all tickers in a specific category"""
    return TICKER_UNIVERSE.get(category, [])

def get_ticker_category(ticker: str) -> str:
    """Get category for a given ticker"""
    for category, tickers in TICKER_UNIVERSE.items():
        if ticker in tickers:
            return category
    return 'Unknown'

def get_total_contracts_estimate() -> int:
    """
    Estimate total contracts to fetch
    
    45 tickers × 252 trading days/year × 5 years × 20 contracts/day
    = 1,134,000 contracts
    """
    num_tickers = len(ALL_TICKERS)
    trading_days_per_year = 252
    years = 5
    contracts_per_day = SAMPLING_CONFIG['contracts_per_day']
    
    return num_tickers * trading_days_per_year * years * contracts_per_day

def get_estimated_storage_size_mb() -> float:
    """
    Estimate storage size in MB
    
    Assumes ~200 bytes per contract after compression
    """
    total_contracts = get_total_contracts_estimate()
    bytes_per_contract = 200  # Compressed Parquet
    
    return (total_contracts * bytes_per_contract) / (1024 * 1024)

# ============================================================================
# CONFIGURATION SUMMARY
# ============================================================================

def print_config_summary():
    """Print configuration summary for verification"""
    print("=" * 70)
    print("THETADATA COLLECTION CONFIGURATION SUMMARY")
    print("=" * 70)
    print(f"\nTicker Universe:")
    for category, tickers in TICKER_UNIVERSE.items():
        print(f"  {category:15s}: {len(tickers):2d} tickers - {', '.join(tickers[:3])}...")
    print(f"  {'TOTAL':15s}: {len(ALL_TICKERS):2d} tickers")
    
    print(f"\nSampling Configuration:")
    print(f"  Expiry Buckets: {SAMPLING_CONFIG['expiry_buckets']}")
    print(f"  Moneyness: {SAMPLING_CONFIG['moneyness_pct']}")
    print(f"  Contracts/Day: {SAMPLING_CONFIG['contracts_per_day']}")
    
    print(f"\nData Range:")
    print(f"  Start: {DATA_RANGE['start_date']}")
    print(f"  End: {DATA_RANGE['end_date']}")
    
    print(f"\nEstimates:")
    print(f"  Total Contracts: {get_total_contracts_estimate():,}")
    print(f"  Storage Size: ~{get_estimated_storage_size_mb():.1f} MB (compressed)")
    
    print(f"\nStorage:")
    print(f"  Raw Parquet: {STORAGE_CONFIG['raw_parquet_dir']}")
    print(f"  Processed CSV: {STORAGE_CONFIG['processed_csv_dir']}")
    
    print("=" * 70)

if __name__ == '__main__':
    # Print config summary when run directly
    print_config_summary()
