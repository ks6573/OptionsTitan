# Data Source Setup (philippdubach/options-data)

## Overview

OptionsTitan uses the **philippdubach/options-data** dataset for free historical options data. No authentication, no API keys, no setup required.

## URLs & Structure

### Base URL

```
https://static.philippdubach.com/data/options
```

### File Layout

| Resource | URL Pattern | Description |
|----------|-------------|-------------|
| Options | `{base}/{ticker}/options.parquet` | Options chain data (parquet) |
| Underlying | `{base}/{ticker}/underlying.parquet` | Underlying OHLC prices |
| Catalog | `{base}/_catalog.parquet` | Available tickers metadata |

### Examples

- **SPY options**: `https://static.philippdubach.com/data/options/spy/options.parquet`
- **AAPL underlying**: `https://static.philippdubach.com/data/options/aapl/underlying.parquet`
- **Catalog**: `https://static.philippdubach.com/data/options/_catalog.parquet`

Tickers are **lowercase** in URLs (e.g., `spy`, `aapl`, `brk.b`).

## Dataset Characteristics

| Property | Value |
|----------|-------|
| **Tickers** | 104 |
| **Date range** | 2008-01-02 to 2025-12-16 |
| **Greeks** | Pre-calculated (delta, gamma, theta, vega, rho) |
| **Implied volatility** | Pre-calculated |
| **Format** | Parquet (columnar, compressed) |

## Local Directories

| Path | Purpose |
|------|---------|
| `data/cache` | Catalog cache, fetcher checkpoints |
| `data/normalized` | Normalized Parquet output (ticker=X/year=Y/) |
| `data/processed_csv` | Legacy CSV output (if used) |

## Query Method

- **Remote query**: DuckDB/Polars reads Parquet URLs directly over HTTPS
- **Predicate pushdown**: Only filtered rows are transferred
- **Lazy scans**: No full file download required

## Optional: Paid Alternatives

If you need data not in the philippdubach dataset (e.g., more tickers, intraday), consider:

- **ThetaData** – Historical options API (subscription)
- **Polygon.io** – Real-time and historical
- **CBOE DataShop** – Exchange data

These require API keys and typically cost $59–199/month. OptionsTitan is designed for the free philippdubach pipeline; paid integrations would require custom adapters.

## References

- **Source**: https://github.com/philippdubach/options-data
- **License**: Educational and research purposes
- **Hosted by**: Philipp Dubach on Cloudflare R2
