"""
OptionsTitan Data Collection Module

Multi-year, multi-ticker historical options data collection using ThetaData REST API.

Components:
- config: Ticker universe and sampling parameters
- thetadata_client: REST API client for ThetaData Terminal
- data_fetcher: Multi-ticker historical data orchestration
- data_normalizer: Schema normalization and multi-year preprocessing
"""

from .config import TICKER_UNIVERSE, SAMPLING_CONFIG, DATA_RANGE

__all__ = [
    'TICKER_UNIVERSE',
    'SAMPLING_CONFIG',
    'DATA_RANGE',
]
