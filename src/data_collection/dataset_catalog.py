"""
Dataset Catalog - Ticker Discovery for philippdubach/options-data

Loads available tickers from _catalog.parquet (remote) with caching.
Falls back to hardcoded list if catalog is unavailable.
"""

import json
import logging
from pathlib import Path
from typing import Optional, Set

from .config import PARQUET_DATASET
from .remote_query import AVAILABLE_TICKERS

logger = logging.getLogger(__name__)

CACHE_TTL_DAYS = 7


def _get_cache_path() -> Path:
    """Path to cached ticker list."""
    cache_dir = Path(PARQUET_DATASET.get('cache_dir', 'data/cache'))
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / 'dataset_tickers.json'


def _load_from_catalog() -> Optional[Set[str]]:
    """Query _catalog.parquet for available tickers. Returns None on failure."""
    try:
        import duckdb
    except ImportError:
        logger.warning("DuckDB not available for catalog query")
        return None

    catalog_url = PARQUET_DATASET.get('catalog_url')
    if not catalog_url:
        return None

    try:
        df = duckdb.query(f"SELECT * FROM read_parquet('{catalog_url}')").to_df()
        if df.empty:
            return None
        # Catalog may have 'ticker', 'symbol', or path-like column
        for col in ('ticker', 'symbol', 'path', 'file'):
            if col in df.columns:
                tickers = df[col].astype(str).str.lower().str.strip().unique()
                # Filter out non-ticker values (paths, etc.)
                valid = {t for t in tickers if t and len(t) <= 10 and t != 'nan'}
                if valid:
                    logger.info("Loaded %d tickers from catalog", len(valid))
                    return valid
        # Fallback: use first string column
        for col in df.columns:
            if df[col].dtype == 'object':
                tickers = df[col].astype(str).str.lower().str.strip().unique()
                valid = {t for t in tickers if t and len(t) <= 10 and t != 'nan'}
                if valid and len(valid) > 10:
                    logger.info("Loaded %d tickers from catalog column '%s'", len(valid), col)
                    return valid
    except Exception as e:
        logger.warning("Catalog query failed: %s", e)

    return None


def get_available_dataset_tickers(force_refresh: bool = False) -> Set[str]:
    """
    Get set of ticker symbols available in the philippdubach dataset.

    Uses catalog parquet when available; caches result for 7 days.
    Falls back to hardcoded list if catalog is unavailable.

    Args:
        force_refresh: If True, ignore cache and re-query catalog.

    Returns:
        Set of lowercase ticker symbols.
    """
    cache_path = _get_cache_path()
    now = __import__('time').time()

    if not force_refresh and cache_path.exists():
        try:
            with open(cache_path) as f:
                data = json.load(f)
            ts = data.get('timestamp', 0)
            if now - ts < CACHE_TTL_DAYS * 86400:
                tickers = set(data.get('tickers', []))
                if tickers:
                    return tickers
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning("Cache read failed: %s", e)

    tickers = _load_from_catalog()
    if tickers is None:
        tickers = set(AVAILABLE_TICKERS)
        logger.info("Using hardcoded ticker list (%d symbols)", len(tickers))

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(cache_path, 'w') as f:
            json.dump({'timestamp': now, 'tickers': sorted(tickers)}, f, indent=0)
    except OSError as e:
        logger.warning("Could not write cache: %s", e)

    return tickers


def has_dataset_coverage(symbol: str) -> bool:
    """
    Check if the given symbol has options history in the philippdubach dataset.

    Args:
        symbol: Stock symbol (case-insensitive).

    Returns:
        True if symbol is in the dataset.
    """
    sym = (symbol or '').strip().lower()
    if not sym:
        return False
    tickers = get_available_dataset_tickers()
    return sym in tickers
