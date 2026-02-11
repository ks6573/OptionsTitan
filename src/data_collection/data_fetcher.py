"""
Historical Data Fetcher - Parquet Pipeline

Fetches multi-year options data from philippdubach/options-data via lazy DuckDB queries.
No full downloads, no subscription APIs. Outputs normalized Parquet (partitioned by ticker/year)
with optional CSV export.
"""

import argparse
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional

import pandas as pd

from .config import (
    PARQUET_DATASET,
    STORAGE_CONFIG,
    QUERY_FILTERS,
)
from .dataset_catalog import get_available_dataset_tickers
from .parquet_query import load_options_slice, load_underlying_slice
from .data_normalizer import MultiYearNormalizer

logger = logging.getLogger(__name__)


def _ensure_dirs():
    """Create cache and normalized directories."""
    for d in (PARQUET_DATASET['cache_dir'], PARQUET_DATASET['normalized_dir']):
        Path(d).mkdir(parents=True, exist_ok=True)


def _checkpoint_path(ticker: str, year: int) -> Path:
    """Path to resumability checkpoint."""
    return Path(PARQUET_DATASET['cache_dir']) / 'fetcher_checkpoints' / f"{ticker}_{year}.json"


def _is_complete(ticker: str, year: int, force_refresh: bool) -> bool:
    """Check if ticker/year partition is already complete."""
    if force_refresh:
        return False
    cp = _checkpoint_path(ticker, year)
    if not cp.exists():
        return False
    try:
        with open(cp) as f:
            return json.load(f).get('complete', False)
    except (json.JSONDecodeError, OSError):
        return False


def _mark_complete(ticker: str, year: int, row_count: int):
    """Mark ticker/year as complete."""
    cp = _checkpoint_path(ticker, year)
    cp.parent.mkdir(parents=True, exist_ok=True)
    with open(cp, 'w') as f:
        json.dump({'complete': True, 'row_count': row_count}, f)


def _fetch_one_ticker_year(
    ticker: str,
    year: int,
    start_date: str,
    end_date: str,
    dte_min: int,
    dte_max: int,
    moneyness_pct: float,
    min_volume: int,
    min_open_interest: int,
    max_spread_pct: Optional[float],
    export_csv: bool,
    force_refresh: bool,
) -> tuple:
    """Fetch and normalize one ticker for one year. Returns (ticker, year, df or None)."""
    if _is_complete(ticker, year, force_refresh=force_refresh):
        logger.info("Skipping %s %d (already complete)", ticker, year)
        return (ticker, year, None)

    y_start = f"{year}-01-01"
    y_end = f"{year}-12-31"
    opts = load_options_slice(
        ticker, y_start, y_end,
        dte_min=dte_min, dte_max=dte_max, moneyness_pct=moneyness_pct,
        min_volume=min_volume, min_open_interest=min_open_interest,
        max_spread_pct=max_spread_pct,
    )
    if opts is None or opts.empty:
        return (ticker, year, None)

    underlying = load_underlying_slice(ticker, y_start, y_end)
    if underlying is not None and not underlying.empty:
        for c in ('close', 'adjusted_close'):
            if c in underlying.columns and 'Close' not in underlying.columns:
                underlying['Close'] = underlying[c]
        if 'volume' in underlying.columns and 'Volume' not in underlying.columns:
            underlying['Volume'] = underlying['volume']
        if 'Date' not in underlying.columns and 'date' in underlying.columns:
            underlying['Date'] = underlying['date']
    else:
        underlying = None

    if 'expiration' in opts.columns and 'expiration_date' not in opts.columns:
        opts = opts.copy()
        opts['expiration_date'] = opts['expiration']

    normalizer = MultiYearNormalizer()
    norm = normalizer.normalize_dataset(opts, ticker.upper(), underlying)
    if norm is None or norm.empty:
        return (ticker, year, None)

    out_dir = Path(PARQUET_DATASET['normalized_dir']) / f"ticker={ticker.upper()}" / f"year={year}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_parquet = out_dir / "part-0.parquet"
    norm.to_parquet(out_parquet, index=False)

    if export_csv:
        csv_dir = Path(STORAGE_CONFIG['processed_csv_dir'])
        csv_dir.mkdir(parents=True, exist_ok=True)
        csv_path = csv_dir / f"{ticker.lower()}_{year}_options.csv"
        norm.to_csv(csv_path, index=False, float_format='%.6f')

    _mark_complete(ticker, year, len(norm))
    return (ticker, year, norm)


def fetch_historical(
    start_date: str = "2019-01-01",
    end_date: str = "2024-12-31",
    tickers: Optional[List[str]] = None,
    force_refresh: bool = False,
    export_csv: bool = False,
    max_workers: int = 8,
    dte_min: int = 30,
    dte_max: int = 60,
    moneyness_pct: float = 0.05,
    min_volume: int = 1,
    min_open_interest: int = 100,
    max_spread_pct: Optional[float] = 0.50,
) -> dict:
    """
    Fetch and normalize options data for multiple tickers and years.

    Returns:
        Dict mapping (ticker, year) -> row count
    """
    _ensure_dirs()
    if tickers is None:
        tickers = sorted(get_available_dataset_tickers())
    tickers = [t.lower().strip() for t in tickers]

    start_year = int(start_date[:4])
    end_year = int(end_date[:4])
    tasks = [(t, y) for t in tickers for y in range(start_year, end_year + 1)]

    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(
                _fetch_one_ticker_year,
                t, y, start_date, end_date,
                dte_min, dte_max, moneyness_pct,
                min_volume, min_open_interest, max_spread_pct,
                export_csv, force_refresh,
            ): (t, y)
            for t, y in tasks
        }
        for future in as_completed(futures):
            t, y = futures[future]
            try:
                _, _, df = future.result()
                results[(t, y)] = len(df) if df is not None else 0
            except Exception as e:
                logger.error("Failed %s %d: %s", t, y, e)
                results[(t, y)] = 0

    return results


def main():
    parser = argparse.ArgumentParser(description="Fetch historical options from philippdubach dataset")
    parser.add_argument("--start", default="2019-01-01", help="Start date")
    parser.add_argument("--end", default="2024-12-31", help="End date")
    parser.add_argument("--tickers", default=None, help="Comma-separated tickers (default: all from catalog)")
    parser.add_argument("--force-refresh", action="store_true", help="Re-fetch even if checkpoint exists")
    parser.add_argument("--export-csv", action="store_true", help="Also export per-ticker CSV")
    parser.add_argument("--workers", type=int, default=8, help="Parallel workers")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    tickers = [t.strip() for t in args.tickers.split(",")] if args.tickers else None
    results = fetch_historical(
        start_date=args.start, end_date=args.end,
        tickers=tickers, force_refresh=args.force_refresh,
        export_csv=args.export_csv, max_workers=args.workers,
    )
    total = sum(results.values())
    print(f"Done. Fetched {total} rows across {len([r for r in results.values() if r > 0])} partitions.")


if __name__ == "__main__":
    main()
