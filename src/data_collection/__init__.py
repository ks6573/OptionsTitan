"""
OptionsTitan Data Collection Module

Multi-year, multi-ticker historical options data collection using FREE open-source data
from philippdubach/options-data (104 tickers, 2008-2025).

Components:
- config: Ticker universe, query filters, and data source configuration
- dataset_catalog: Ticker discovery from _catalog.parquet with caching
- parquet_query: load_options_slice, load_underlying_slice (lazy DuckDB)
- remote_query: DuckDB/Polars remote Parquet queries (predicate pushdown)
- schema_contract: Schema validation and drift detection
- data_fetcher: Multi-ticker Parquet pipeline with resumability
- data_normalizer: Schema normalization and feature engineering
"""

from .config import (
    TICKER_UNIVERSE,
    SAMPLING_CONFIG,
    DATA_RANGE,
    PARQUET_CONFIG,
    PARQUET_DATASET,
    QUERY_FILTERS,
)
from .dataset_catalog import get_available_dataset_tickers, has_dataset_coverage

__all__ = [
    'TICKER_UNIVERSE',
    'SAMPLING_CONFIG',
    'DATA_RANGE',
    'PARQUET_CONFIG',
    'PARQUET_DATASET',
    'QUERY_FILTERS',
    'get_available_dataset_tickers',
    'has_dataset_coverage',
]
