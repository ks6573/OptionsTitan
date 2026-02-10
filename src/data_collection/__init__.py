"""
OptionsTitan Data Collection Module

Multi-year, multi-ticker historical options data collection using FREE open-source data
from philippdubach/options-data (104 tickers, 2008-2025).

Components:
- config: Ticker universe, query filters, and data source configuration
- remote_query: DuckDB/Polars remote Parquet queries (predicate pushdown)
- schema_contract: Schema validation and drift detection
- corporate_actions: Stock split validation and continuity checks
- data_fetcher: Multi-ticker data orchestration (legacy - can use remote_query directly)
- data_normalizer: Schema normalization and feature engineering
"""

from .config import (
    TICKER_UNIVERSE, 
    SAMPLING_CONFIG, 
    DATA_RANGE,
    PARQUET_CONFIG,
    QUERY_FILTERS,
)

__all__ = [
    'TICKER_UNIVERSE',
    'SAMPLING_CONFIG',
    'DATA_RANGE',
    'PARQUET_CONFIG',
    'QUERY_FILTERS',
]
