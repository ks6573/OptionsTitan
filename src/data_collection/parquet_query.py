"""
Parquet Query - Spec-exact API for lazy options/underlying slices.

Wraps remote_query with load_options_slice and load_underlying_slice.
Supports moneyness filtering (post-query with underlying merge) and max_spread_pct.
"""

import logging
from typing import Optional

import pandas as pd

from .remote_query import (
    query_remote_parquet,
    query_underlying_prices,
)

logger = logging.getLogger(__name__)


def load_options_slice(
    symbol: str,
    start_date: str,
    end_date: str,
    dte_min: int,
    dte_max: int,
    moneyness_pct: float,
    min_volume: int = 1,
    min_open_interest: int = 0,
    max_spread_pct: Optional[float] = None,
) -> pd.DataFrame:
    """
    Load options data slice with filters. Uses lazy DuckDB query (no full download).

    Args:
        symbol: Stock symbol (case-insensitive)
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        dte_min: Minimum days to expiration
        dte_max: Maximum days to expiration
        moneyness_pct: Max moneyness e.g. 0.05 for ATM +/- 5%
        min_volume: Minimum option volume
        min_open_interest: Minimum open interest
        max_spread_pct: Max bid-ask spread as % of mid (None = no filter)

    Returns:
        DataFrame with options data plus computed: option_price, strike_distance,
        time_to_expiry, strike_distance_pct. Moneyness filtered.
    """
    filters = {
        'date_min': start_date,
        'date_max': end_date,
        'dte_min': dte_min,
        'dte_max': dte_max,
        'min_volume': min_volume,
        'min_open_interest': min_open_interest,
    }

    opts = query_remote_parquet(symbol, filters, engine='duckdb')
    if opts is None or opts.empty:
        return pd.DataFrame()

    opts = opts.copy()

    # Need underlying for moneyness filter
    underlying = query_underlying_prices(symbol, date_min=start_date, date_max=end_date)
    if underlying is not None and not underlying.empty:
        for c in ('close', 'adjusted_close', 'Close'):
            if c in underlying.columns:
                spot_col = c
                break
        else:
            spot_col = None
        if spot_col:
            under = underlying[['date', spot_col]].copy()
            under.columns = ['_merge_date', 'spot']
            under['_merge_date'] = pd.to_datetime(under['_merge_date']).dt.normalize()
            opts['_merge_date'] = pd.to_datetime(opts['date']).dt.normalize()
            opts = opts.merge(under, on='_merge_date', how='left')
            opts = opts.drop(columns=['_merge_date'], errors='ignore')
            opts['spot'] = pd.to_numeric(opts['spot'], errors='coerce').fillna(0)

            # Moneyness filter: |strike - spot| / spot <= moneyness_pct
            opts = opts[opts['spot'] > 0].copy()
            opts['_m'] = (opts['strike'] - opts['spot']) / opts['spot']
            opts = opts[opts['_m'].abs() <= moneyness_pct].drop(columns=['_m'])

    # option_price: mark > last > mid
    if 'mark' in opts.columns:
        opts['option_price'] = opts['mark']
    elif 'last' in opts.columns:
        opts['option_price'] = opts['last']
    elif 'bid' in opts.columns and 'ask' in opts.columns:
        opts['option_price'] = (opts['bid'] + opts['ask']) / 2.0
    else:
        opts['option_price'] = 0.0

    # strike_distance (dollars) and time_to_expiry
    if 'spot' in opts.columns:
        opts['strike_distance'] = opts['strike'] - opts['spot']
        opts['strike_distance_pct'] = opts['strike_distance'] / opts['spot']
    else:
        opts['strike_distance'] = 0.0
        opts['strike_distance_pct'] = 0.0

    opts['expiration'] = pd.to_datetime(opts['expiration'])
    opts['date'] = pd.to_datetime(opts['date'])
    opts['time_to_expiry'] = (opts['expiration'] - opts['date']).dt.days

    # max_spread_pct filter
    if max_spread_pct is not None and 'bid' in opts.columns and 'ask' in opts.columns:
        mid = (opts['bid'] + opts['ask']) / 2.0
        spread_pct = (opts['ask'] - opts['bid']) / mid.replace(0, 1e-10)
        opts = opts[spread_pct <= max_spread_pct]

    return opts.reset_index(drop=True)


def load_underlying_slice(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Load underlying price slice.

    Args:
        symbol: Stock symbol
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)

    Returns:
        DataFrame with date, close (or adjusted_close), and other OHLCV columns.
    """
    df = query_underlying_prices(symbol, date_min=start_date, date_max=end_date)
    if df is None:
        return pd.DataFrame()
    df = df.copy()
    # Normalize column names for downstream
    if 'close' not in df.columns and 'adjusted_close' in df.columns:
        df['close'] = df['adjusted_close']
    elif 'Close' in df.columns and 'close' not in df.columns:
        df['close'] = df['Close']
    return df
