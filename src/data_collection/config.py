"""
Data Collection Configuration

Defines the ticker universe, sampling parameters, and date ranges for
multi-year historical options data collection.

**MIGRATED TO FREE DATA SOURCE (philippdubach/options-data)**
- 104 tickers available (2008-2025)
- Remote query with predicate pushdown (no full downloads)
- Pre-calculated Greeks and implied volatility
- Zero cost, zero authentication, zero setup

Based on GPT-5 recommendations for production-grade data collection.
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
# PARQUET DATASET CONFIGURATION (PRIMARY - philippdubach/options-data)
# ============================================================================

PARQUET_CONFIG: Dict = {
    # Base URL for philippdubach dataset (Cloudflare R2)
    'base_url': 'https://static.philippdubach.com/data/options',
    
    # Dataset metadata
    'dataset_snapshot_date': '2025-12-16',  # Last update of dataset
    'dataset_version': 'v1',
    'total_tickers': 104,
    'date_range': ('2008-01-02', '2025-12-16'),
    
    # Available tickers (104 total - lowercase)
    'available_tickers': [
        "aapl", "abbv", "abt", "acn", "adbe", "aig", "amd", "amgn", "amt", "amzn",
        "avgo", "axp", "ba", "bac", "bk", "bkng", "blk", "bmy", "brk.b", "c",
        "cat", "cl", "cmcsa", "cof", "cop", "cost", "crm", "csco", "cvs", "cvx",
        "de", "dhr", "dis", "duk", "emr", "fdx", "gd", "ge", "gild", "gm",
        "goog", "googl", "gs", "hd", "hon", "ibm", "intu", "isrg", "iwm", "jnj",
        "jpm", "ko", "lin", "lly", "lmt", "low", "ma", "mcd", "mdlz", "mdt",
        "met", "meta", "mmm", "mo", "mrk", "ms", "msft", "nee", "nflx", "nke",
        "now", "nvda", "orcl", "pep", "pfe", "pg", "pltr", "pm", "pypl", "qcom",
        "qqq", "rtx", "sbux", "schw", "so", "spg", "spy", "t", "tgt", "tmo",
        "tmus", "tsla", "txn", "uber", "unh", "unp", "ups", "usb", "v", "vix",
        "vz", "wfc", "wmt", "xom"
    ],
    
    # Optional catalog file (if available)
    'catalog_url': 'https://static.philippdubach.com/data/options/_catalog.parquet',
}

# Alias for pipeline compatibility (cache_dir, normalized_dir)
PARQUET_DATASET: Dict = {
    'base_url': 'https://static.philippdubach.com/data/options',
    'catalog_url': 'https://static.philippdubach.com/data/options/_catalog.parquet',
    'cache_dir': 'data/cache',
    'normalized_dir': 'data/normalized',
}

# ============================================================================
# SAMPLING MODES (Training Pipeline)
# ============================================================================

SAMPLING_MODES: Dict = {
    'MODE_A': {  # Default: 30-60 DTE, ATM +/- 5%, liquidity filtered
        'dte_min': 30,
        'dte_max': 60,
        'moneyness_pct': 0.05,
        'min_volume': 1,
        'min_open_interest': 100,
    },
    'MODE_B': {  # Richer: multiple DTE buckets, +/- 10% moneyness
        'dte_min': 7,
        'dte_max': 180,
        'moneyness_pct': 0.10,
        'min_volume': 1,
        'min_open_interest': 100,
    },
}

# ============================================================================
# REMOTE QUERY FILTERS (Predicate Pushdown Configuration)
# ============================================================================

QUERY_FILTERS: Dict = {
    # Days to Expiration (DTE) filters
    'dte_min': 25,   # Minimum DTE (e.g., ~1 month)
    'dte_max': 75,   # Maximum DTE (e.g., ~2.5 months)
    
    # Moneyness filters (strike relative to spot)
    # -0.05 = 5% OTM, +0.05 = 5% ITM
    'moneyness_min': -0.05,
    'moneyness_max': 0.05,
    
    # Liquidity filters (minimum thresholds)
    'min_volume': 1,              # Require at least some volume
    'min_open_interest': 100,     # Require meaningful open interest
    
    # Microstructure quality filters
    'max_spread_pct': 0.50,      # Max 50% bid-ask spread
    'min_bid': 0.01,             # Minimum bid price (avoid $0 bids)
    'min_ask': 0.01,             # Minimum ask price
    
    # Option type filter (None = both calls and puts)
    'option_type': None,  # 'call', 'put', or None for both
}

# ============================================================================
# KNOWN STOCK SPLITS (Corporate Actions Catalog)
# ============================================================================

STOCK_SPLITS: Dict[str, List[Tuple[str, float, str]]] = {
    # Format: ticker: [(date, ratio, description), ...]
    "aapl": [
        ("2020-08-31", 4.0, "4-for-1 split"),
        ("2014-06-09", 7.0, "7-for-1 split"),
    ],
    "tsla": [
        ("2022-08-25", 3.0, "3-for-1 split"),
        ("2020-08-31", 5.0, "5-for-1 split"),
    ],
    "nvda": [
        ("2024-06-10", 10.0, "10-for-1 split"),
        ("2021-07-20", 4.0, "4-for-1 split"),
    ],
    "googl": [
        ("2022-07-18", 20.0, "20-for-1 split"),
    ],
    "goog": [
        ("2022-07-18", 20.0, "20-for-1 split"),
    ],
    "amzn": [
        ("2022-06-06", 20.0, "20-for-1 split"),
    ],
}


# ============================================================================
# STORAGE CONFIGURATION
# ============================================================================

STORAGE_CONFIG: Dict = {
    # Directory structure
    'raw_parquet_dir': 'data/raw_parquet',
    'processed_csv_dir': 'data/processed_csv',
    'metadata_dir': 'data/metadata',
    'cache_dir': 'data/cache',
    'normalized_dir': 'data/normalized',
    
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
    'strike_distance',    # strike - underlying_price (DOLLARS, not percent)
    'time_to_expiry',     # Days to expiration (DTE)
    'volume',             # Option volume
    'implied_volatility', # IV (pre-calculated in philippdubach dataset)
    'vix_level',          # VIX index level
    'spy_return_5min',    # Daily SPY return proxy (Training.py compat; no intraday)
    'rsi',                # RSI indicator on underlying
    'timestamp',          # YYYY-MM-DD HH:MM:SS
]

# spy_return_5min: populated with daily SPY close-to-close return (proxy for intraday).
# Training.py expects this column name. Schema contract accepts spy_return_1d as alias.

# Additional columns to capture (not required by Training.py but useful)
OPTIONAL_COLUMNS: List[str] = [
    'ticker',             # Stock symbol
    'sector',             # Sector classification
    'contract_symbol',    # Full option symbol (OCC format)
    'contract_id',        # Unique contract identifier (philippdubach)
    'expiration_date',    # Option expiration date
    'strike',             # Strike price
    'option_type',        # 'C' or 'P' (or 'call'/'put' in philippdubach)
    'bid',                # Bid price
    'ask',                # Ask price
    'mid',                # Mid price (bid + ask) / 2
    'mark',               # Mark price (philippdubach dataset)
    'spread',             # Bid-ask spread (ask - bid)
    'spread_pct',         # Spread as % of mid
    'bid_size',           # Bid size
    'ask_size',           # Ask size
    'open_interest',      # Open interest
    'open',               # Option open price
    'high',               # Option high price
    'low',                # Option low price
    'last',               # Last traded price (philippdubach)
    'underlying_volume',  # Underlying stock volume
    'date',               # Calendar date (YYYY-MM-DD)
    'delta',              # Option delta (philippdubach)
    'gamma',              # Option gamma (philippdubach)
    'theta',              # Option theta (philippdubach)
    'vega',               # Option vega (philippdubach)
    'rho',                # Option rho (philippdubach)
    'in_the_money',       # ITM flag (philippdubach)
    'moneyness',          # strike / spot (calculated)
    'strike_distance_pct', # (strike - spot) / spot (calculated)
]

# ============================================================================
# PHILIPPDUBACH DATASET SCHEMA
# ============================================================================

PHILIPPDUBACH_SCHEMA: Dict[str, str] = {
    # Expected columns and dtypes from philippdubach/options-data
    "contract_id": "object",
    "symbol": "object",
    "expiration": "datetime64[ns]",
    "strike": "float64",
    "type": "object",  # "call" or "put"
    "last": "float64",
    "mark": "float64",
    "bid": "float64",
    "ask": "float64",
    "bid_size": "int64",
    "ask_size": "int64",
    "volume": "int64",
    "open_interest": "int64",
    "date": "datetime64[ns]",
    "implied_volatility": "float64",
    "delta": "float64",
    "gamma": "float64",
    "theta": "float64",
    "vega": "float64",
    "rho": "float64",
    "in_the_money": "bool",
}

# ============================================================================
# DATASET METADATA (Snapshot Tracking)
# ============================================================================

DATASET_METADATA: Dict = {
    # Track which version of the dataset was used for training
    "source": "philippdubach/options-data",
    "snapshot_date": "2025-12-16",
    "url": "https://github.com/philippdubach/options-data",
    "license": "Educational and research purposes",
    
    # Ticker universe actually used
    "tickers_used": None,  # Will be populated at runtime
    "ticker_count": None,
    
    # Date range actually queried
    "date_range_start": None,
    "date_range_end": None,
    
    # Row counts
    "total_contracts_fetched": None,
    "total_contracts_after_filters": None,
    
    # Query metadata
    "query_filters_applied": None,
    "corporate_actions_validated": None,
    
    # Execution metadata
    "collection_timestamp": None,
    "collection_duration_seconds": None,
}

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
    print("OPTIONS DATA COLLECTION CONFIGURATION SUMMARY")
    print("=" * 70)
    print(f"\n📊 PRIMARY DATA SOURCE: {DATASET_METADATA['source']}")
    print(f"   Snapshot Date: {PARQUET_CONFIG['dataset_snapshot_date']}")
    print(f"   Available Tickers: {PARQUET_CONFIG['total_tickers']}")
    print(f"   Date Coverage: {PARQUET_CONFIG['date_range'][0]} to {PARQUET_CONFIG['date_range'][1]}")
    print(f"   Cost: FREE (no authentication required)")
    
    print(f"\nTicker Universe (Original 45):")
    for category, tickers in TICKER_UNIVERSE.items():
        print(f"  {category:15s}: {len(tickers):2d} tickers - {', '.join(tickers[:3])}...")
    print(f"  {'TOTAL':15s}: {len(ALL_TICKERS):2d} tickers")
    
    print(f"\nQuery Filters (Predicate Pushdown):")
    print(f"  DTE Range: {QUERY_FILTERS['dte_min']}-{QUERY_FILTERS['dte_max']} days")
    print(f"  Moneyness: {QUERY_FILTERS['moneyness_min']*100:.0f}% to {QUERY_FILTERS['moneyness_max']*100:.0f}%")
    print(f"  Min Volume: {QUERY_FILTERS['min_volume']}")
    print(f"  Min OI: {QUERY_FILTERS['min_open_interest']}")
    print(f"  Max Spread: {QUERY_FILTERS['max_spread_pct']*100:.0f}%")
    
    print(f"\nData Range (Query Target):")
    print(f"  Start: {DATA_RANGE['start_date']}")
    print(f"  End: {DATA_RANGE['end_date']}")
    
    print(f"\nCorporate Actions Tracking:")
    splits_count = sum(len(events) for events in STOCK_SPLITS.values())
    print(f"  Tickers with known splits: {len(STOCK_SPLITS)}")
    print(f"  Total split events tracked: {splits_count}")
    
    print(f"\nRequired Schema (Training.py):")
    print(f"  Columns: {len(REQUIRED_COLUMNS)}")
    print(f"  Note: spy_return_5min → spy_return_1d (daily proxy)")
    
    print(f"\nStorage:")
    print(f"  Raw Parquet: {STORAGE_CONFIG['raw_parquet_dir']}")
    print(f"  Processed CSV: {STORAGE_CONFIG['processed_csv_dir']}")
    print(f"  Metadata: {STORAGE_CONFIG['metadata_dir']}")
    
    print(f"\n💡 Benefits:")
    print(f"  ✓ Zero cost (FREE open-source dataset)")
    print(f"  ✓ Zero setup (no API keys, no authentication)")
    print(f"  ✓ Pre-calculated Greeks and IV")
    print(f"  ✓ Remote query (no full dataset download)")
    print(f"  ✓ 104 tickers available (vs. 45 planned)")
    print(f"  ✓ Longer history (2008 vs. 2019)")
    
    print("=" * 70)

if __name__ == '__main__':
    # Print config summary when run directly
    print_config_summary()
