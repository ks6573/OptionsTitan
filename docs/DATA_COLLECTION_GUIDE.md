# Historical Options Data Collection Guide

## Overview

This guide walks you through collecting free historical options data using the philippdubach/options-data dataset (104 tickers, 2008-2025) with zero setup required.

## Why This Is Better

**Free Alternative to Paid Services:**

| Feature | Paid Services | philippdubach (Current) |
|---------|---------------|-------------------------|
| Cost | $59-199/month | **FREE** |
| Setup | Terminal + Auth | **None** |
| Tickers | 45 planned | **104 available** |
| History | 2019-2024 | **2008-2025** |
| Collection Time | 8-12 hours | **5-10 minutes** |
| Greeks/IV | Manual calc | **Pre-calculated** |
| Authentication | Required | **None** |

## Prerequisites

Before starting:

1. ✅ **Python 3.9+** installed
2. ✅ **DuckDB** installed: `pip install duckdb` or `uv sync`
3. ✅ **Internet connection** for remote queries

**That's it!** No API keys, no Terminal installation, no authentication.

---

## Quick Start

### Test the Infrastructure

```bash
# Run validation suite
python test_free_data_migration.py

# Expected: 6/6 tests passing
```

### Query Data Directly

```python
from src.data_collection import remote_query

# Query SPY options (Q4 2024, 30-60 DTE)
filters = {
    "date_min": "2024-10-01",
    "date_max": "2024-12-31",
    "dte_min": 30,
    "dte_max": 60,
    "min_volume": 1,
    "min_open_interest": 100,
}

df = remote_query.query_remote_parquet("spy", filters)
print(f"Retrieved {len(df):,} contracts")
```

**No downloads required** - data is queried remotely with predicate pushdown!

---

## Parquet Dataset Workflow

### Catalog & Lazy Queries

OptionsTitan uses the philippdubach dataset with **lazy queries** (no full downloads):

1. **Dataset Catalog** (`src/data_collection/dataset_catalog.py`)
   - `get_available_dataset_tickers()`: Loads tickers from `_catalog.parquet` or cached JSON
   - `has_dataset_coverage(symbol)`: Check if a symbol has options history
   - Cache: `data/cache/dataset_tickers.json` (7-day TTL)

2. **Parquet Query Layer** (`src/data_collection/parquet_query.py`)
   - `load_options_slice()`: Filter by date, DTE, moneyness, liquidity
   - `load_underlying_slice()`: Underlying OHLC for a date range
   - Thin wrapper over `remote_query` with schema-aware filters

3. **Normalized Output**
   - Data fetcher writes to `data/normalized/ticker={TICKER}/year={YYYY}/part-*.parquet`
   - Schema: `REQUIRED_COLUMNS` (price, option_price, strike_distance, etc.)
   - Training pipeline prefers `data/normalized` over `data/processed_csv`

### Data Fetcher CLI

Collect normalized data without full downloads:

```bash
python -m src.data_collection.data_fetcher --start 2019-01-01 --end 2024-12-31

# Specific tickers
python -m src.data_collection.data_fetcher --start 2019-01-01 --end 2024-12-31 --tickers SPY,AAPL,TSLA

# Force refresh (ignore checkpoints)
python -m src.data_collection.data_fetcher --start 2019-01-01 --end 2024-12-31 --force-refresh

# Optional: export CSV
python -m src.data_collection.data_fetcher --start 2019-01-01 --end 2024-12-31 --export-csv
```

Resumability: checkpoints in `data/cache/fetcher_checkpoints/`; skip completed ticker/year unless `--force-refresh`.

---

## Available Tickers (104 Total)

The dataset includes all major US equities and ETFs:

### ETFs & Indices
SPY, QQQ, IWM, DIA, XLF, XLK, XLE, VIX

### Mega-Cap Tech
AAPL, MSFT, NVDA, AMZN, GOOGL, META, TSLA, AMD, NFLX, ORCL, CRM, INTU

### Financials
JPM, BAC, GS, MS, WFC, SCHW, BK, AXP, BLK, MET

### Healthcare
UNH, JNJ, PFE, ABBV, LLY, ABT, TMO, MDT, BMY, GILD

### Energy
XOM, CVX, COP, SLB

### Consumer
WMT, COST, TGT, DIS, NKE, MCD, SBUX, KO, PEP

### And many more...

[Full list from `get_available_dataset_tickers()` in src/data_collection/dataset_catalog.py](../src/data_collection/dataset_catalog.py)

---

## Data Collection Methods

### Method 1: Remote Query (Recommended)

**Query data on-demand without downloading full files:**

```python
from src.data_collection import remote_query

# Query multiple tickers in parallel
tickers = ["spy", "aapl", "tsla"]
filters = {
    "date_min": "2024-01-01",
    "date_max": "2024-12-31",
    "dte_min": 30,
    "dte_max": 60,
    "min_volume": 1,
    "min_open_interest": 100,
}

results = remote_query.batch_query_tickers(
    tickers,
    filters,
    max_workers=10
)

for ticker, df in results.items():
    print(f"{ticker}: {len(df):,} contracts")
```

**Benefits:**
- Only downloads filtered data (10-100x bandwidth savings)
- No local storage of full dataset needed
- Parallel queries across multiple tickers
- Instant results

### Method 2: Direct Download (If Needed)

**Download full Parquet files for offline use:**

```python
import urllib.request

ticker = "spy"  # lowercase
urllib.request.urlretrieve(
    f"https://static.philippdubach.com/data/options/{ticker}/options.parquet",
    f"data/raw/{ticker}_options.parquet"
)
```

**Use when:**
- Need offline access
- Want to experiment with full dataset locally
- Building custom queries

---

## Query Filters Explained

### Date Filters

```python
"date_min": "2024-01-01",  # Start date
"date_max": "2024-12-31",  # End date
```

### Days to Expiration (DTE)

```python
"dte_min": 30,  # Minimum 30 days to expiration
"dte_max": 60,  # Maximum 60 days to expiration
```

**Common DTE ranges:**
- 25-35 days: ~1 month (standard monthly)
- 45-75 days: ~2 months (quarterly)
- 7-21 days: Weekly options
- 60-120 days: Long-term options

### Moneyness (Strike vs Spot)

```python
"moneyness_min": -0.05,  # 5% OTM
"moneyness_max": 0.05,   # 5% ITM
```

**Moneyness ranges:**
- -0.05 to 0.05: Near the money (±5%)
- -0.10 to 0.10: Moderate range (±10%)
- -0.20 to 0.20: Wide range (±20%)

### Liquidity Filters

```python
"min_volume": 1,           # Minimum daily volume
"min_open_interest": 100,  # Minimum open interest
```

**Recommended thresholds:**
- High liquidity: volume > 100, OI > 1000
- Medium liquidity: volume > 10, OI > 100
- Include all: volume > 0, OI > 0

---

## Validation & Quality Checks

### Schema Validation

```python
from src.data_collection import schema_contract

# Validate data matches expected schema
report = schema_contract.full_validation_report(
    df,
    schema_type="philippdubach",
    check_uniqueness_keys=["contract_id", "date"]
)

if report['overall_valid']:
    print("✓ All validations passed")
else:
    print(f"Errors: {report['errors']}")
```

### Corporate Action Validation

```python
from src.data_collection import corporate_actions

# Check for stock splits
splits = corporate_actions.get_splits_for_ticker("aapl")
print(f"Known splits: {splits}")

# Validate split handling
is_valid, anomalies = corporate_actions.validate_split_continuity(
    options_df, 
    "aapl", 
    underlying_df
)
```

**Tracked splits:**
- AAPL: 4:1 (2020), 7:1 (2014)
- TSLA: 3:1 (2022), 5:1 (2020)
- NVDA: 10:1 (2024), 4:1 (2021)
- GOOGL: 20:1 (2022)
- AMZN: 20:1 (2022)

---

## Advanced Usage

### Multi-Year Collection

```python
# Collect data across multiple years
filters = {
    "date_min": "2019-01-01",
    "date_max": "2024-12-31",
    "dte_min": 25,
    "dte_max": 75,
    "min_volume": 1,
    "min_open_interest": 100,
}

# Query all 104 tickers
from src.data_collection.remote_query import AVAILABLE_TICKERS

results = remote_query.batch_query_tickers(
    AVAILABLE_TICKERS,
    filters,
    max_workers=20  # Parallel queries
)
```

**Performance:**
- 104 tickers in ~5-10 minutes
- ~100-500 MB total bandwidth
- ~200-500 MB local storage

### Microstructure Filters

Add quality filters for better signal:

```python
# After querying, apply quality filters
df['mid'] = (df['bid'] + df['ask']) / 2
df['spread'] = df['ask'] - df['bid']
df['spread_pct'] = df['spread'] / df['mid']

# Filter out poor quality data
df_clean = df[
    (df['spread_pct'] <= 0.50) &  # Max 50% spread
    (df['bid'] > 0) &
    (df['ask'] > 0) &
    (df['volume'] > 0)
]
```

### Multi-Era Validation

Guard against survivorship bias:

```python
eras = [
    ("2008-2012", "Financial Crisis & Recovery"),
    ("2013-2017", "Bull Market"),
    ("2018-2021", "COVID Era"),
    ("2022-2024", "Rate Hikes"),
]

for (start, end), description in eras:
    filters['date_min'] = start
    filters['date_max'] = end
    
    df = remote_query.query_remote_parquet(ticker, filters)
    print(f"{description}: {len(df):,} contracts")
```

---

## Data Schema

### Options Data (philippdubach)

The dataset includes all these pre-calculated fields:

```
contract_id       - Unique identifier
symbol            - Ticker symbol
expiration        - Expiration date
strike            - Strike price
type              - "call" or "put"
last              - Last traded price
mark              - Mark price (mid)
bid, ask          - Bid/ask prices
bid_size, ask_size - Market depth
volume            - Daily volume
open_interest     - Open interest
date              - Quote date
implied_volatility - Black-Scholes IV
delta, gamma, theta, vega, rho - Greeks
in_the_money      - ITM flag
```

### Underlying Prices

```
symbol            - Ticker
date              - Trading date
open, high, low, close - OHLC
adjusted_close    - Adjusted for splits
volume            - Stock volume
dividend_amount   - Dividend if any
split_coefficient - Split ratio if any
```

---

## Training Pipeline Integration

### Schema Transformation

The data normalizer transforms philippdubach → Training.py format:

```python
# philippdubach schema → Training.py required columns
mark (or mid)              → option_price
strike                     → strike_distance (calculated)
(expiration - date).days   → time_to_expiry
volume                     → volume
implied_volatility         → implied_volatility
underlying close           → price
VIX close                  → vix_level
daily SPY close-to-close   → spy_return_5min (daily proxy; Training.py expects this column)
RSI(14)                    → rsi
date                       → timestamp
```

### Running Training

After collecting data:

```bash
# Single-ticker training (original)
python src/Training.py

# Multi-ticker training (recommended)
python src/Training_MultiTicker.py
```

---

## Troubleshooting

### Issue: "DuckDB not available"

**Solution:**
```bash
uv sync
# OR
pip install duckdb
```

### Issue: "No data retrieved"

**Check:**
1. Date range within 2008-2025
2. Ticker is lowercase (e.g., "spy" not "SPY")
3. Ticker in dataset: `from src.data_collection.dataset_catalog import has_dataset_coverage; has_dataset_coverage("spy")`
4. Internet connection working

**Debug:**
```python
# Check file exists
from src.data_collection import remote_query
exists = remote_query.validate_file_exists(
    remote_query.build_options_url("spy")
)
print(f"File exists: {exists}")
```

### Issue: "Empty DataFrame"

**Cause:** Filters too restrictive (no contracts match)

**Solution:**
```python
# Relax filters
filters = {
    "date_min": "2024-01-01",
    "date_max": "2024-12-31",
    "min_volume": 0,        # Accept any volume
    "min_open_interest": 0, # Accept any OI
}

# Or estimate before querying
row_count = remote_query.estimate_filtered_size("spy", filters)
print(f"Estimated rows: {row_count:,}")
```

### Issue: Schema validation errors

**Cause:** Upstream dataset changed

**Solution:**
```python
# Check for schema drift
from src.data_collection import schema_contract

drift = schema_contract.detect_schema_drift(df, schema_contract.PHILIPPDUBACH_SCHEMA)
if drift:
    print(f"New columns in dataset: {drift}")
```

---

## Performance Comparison

### Collection Speed

| Tickers | Method | Time | Bandwidth |
|---------|--------|------|-----------|
| 1 ticker | Remote query | ~3 seconds | ~1-10 MB |
| 10 tickers | Batch query | ~30 seconds | ~50-100 MB |
| 104 tickers | Full batch | ~5-10 minutes | ~100-500 MB |

### vs. Paid Services

**Paid services (typical):**
- Cost: $59-199/month
- Setup: 10-30 minutes (auth, API keys)
- Collection: 8-12 hours for 45 tickers
- Bandwidth: 1-2 GB
- Rate limited: 3 requests/second

**After (philippdubach):**
- Cost: **$0**
- Setup: **0 minutes**
- Collection: **5-10 minutes for 104 tickers**
- Bandwidth: **100-500 MB**
- Rate limits: **None**

---

## Best Practices

### 1. Start Small, Scale Up

```python
# Test with 1 ticker first
df_spy = remote_query.query_remote_parquet("spy", filters)

# Then expand to a few
tickers = ["spy", "aapl", "tsla"]
results = remote_query.batch_query_tickers(tickers, filters)

# Finally, query all 104
from src.data_collection.remote_query import AVAILABLE_TICKERS
all_results = remote_query.batch_query_tickers(AVAILABLE_TICKERS, filters, max_workers=20)
```

### 2. Validate Early and Often

```python
# Always validate after querying
from src.data_collection import schema_contract

report = schema_contract.full_validation_report(df, "philippdubach")
assert report['overall_valid'], f"Validation failed: {report['errors']}"
```

### 3. Use Appropriate Filters

```python
# For training ML models - tight filters
filters = {
    "dte_min": 30,
    "dte_max": 60,
    "moneyness_min": -0.05,
    "moneyness_max": 0.05,
    "min_volume": 10,
    "min_open_interest": 500,
    "max_spread_pct": 0.30,
}

# For exploration - loose filters
filters = {
    "date_min": "2024-01-01",
    "date_max": "2024-12-31",
    "min_volume": 0,
    "min_open_interest": 0,
}
```

### 4. Cache Results Locally

```python
import pandas as pd

# Query once
df = remote_query.query_remote_parquet("aapl", filters)

# Save locally
df.to_parquet("data/cache/aapl_2024.parquet")

# Load from cache later
df_cached = pd.read_parquet("data/cache/aapl_2024.parquet")
```

---

## Dataset Coverage

### Date Range
- **Start**: January 2, 2008
- **End**: December 16, 2025
- **Total**: 17+ years of data

### Market Regimes Covered
- ✅ 2008 Financial Crisis
- ✅ 2009-2019 Bull Market
- ✅ 2020 COVID Crash & Recovery
- ✅ 2022-2023 Rate Hikes
- ✅ 2024-2025 Current Market

### Data Fields
- ✅ All Greeks (delta, gamma, theta, vega, rho) - **pre-calculated**
- ✅ Implied volatility - **pre-calculated**
- ✅ Bid/ask prices and sizes
- ✅ Volume and open interest
- ✅ Underlying prices (OHLC, adjusted)

---

## Examples

### Example 1: Collect Training Data

```python
from src.data_collection import remote_query, schema_contract

# Define training criteria
filters = {
    "date_min": "2019-01-01",
    "date_max": "2024-12-31",
    "dte_min": 25,
    "dte_max": 75,
    "min_volume": 1,
    "min_open_interest": 100,
}

# Collect for key tickers
tickers = ["spy", "qqq", "iwm", "aapl", "msft", "nvda"]

for ticker in tickers:
    print(f"\nCollecting {ticker.upper()}...")
    
    # Query options
    df = remote_query.query_remote_parquet(ticker, filters)
    
    # Validate
    report = schema_contract.full_validation_report(df, "philippdubach")
    
    if report['overall_valid']:
        # Save to disk
        df.to_parquet(f"data/training/{ticker}_options.parquet")
        print(f"✓ Saved {len(df):,} contracts")
    else:
        print(f"✗ Validation failed: {report['errors']}")
```

### Example 2: Validate Corporate Actions

```python
from src.data_collection import remote_query, corporate_actions

# Query data around NVDA 2024 split (2024-06-10, 10:1)
filters = {
    "date_min": "2024-01-01",
    "date_max": "2024-12-31",
    "dte_min": 30,
    "dte_max": 60,
}

options_df = remote_query.query_remote_parquet("nvda", filters)
underlying_df = remote_query.query_underlying_prices(
    "nvda",
    date_min="2024-01-01",
    date_max="2024-12-31"
)

# Run validation
results = corporate_actions.full_corporate_action_validation(
    options_df,
    underlying_df,
    "nvda",
    save_report=True
)

print(f"Validation: {'PASSED' if results['overall_valid'] else 'FAILED'}")
```

### Example 3: Multi-Era Analysis

```python
# Compare data quality across eras
eras = {
    "pre_covid": ("2019-01-01", "2020-02-28"),
    "covid": ("2020-03-01", "2021-12-31"),
    "rate_hikes": ("2022-01-01", "2023-12-31"),
    "current": ("2024-01-01", "2024-12-31"),
}

for era_name, (start, end) in eras.items():
    filters = {
        "date_min": start,
        "date_max": end,
        "dte_min": 30,
        "dte_max": 60,
    }
    
    df = remote_query.query_remote_parquet("spy", filters)
    
    avg_iv = df['implied_volatility'].mean()
    avg_volume = df['volume'].mean()
    
    print(f"{era_name}: {len(df):,} contracts, IV={avg_iv:.2f}, Vol={avg_volume:.0f}")
```

---

## Support & Resources

### Documentation
- [README_DATA_COLLECTION.md](../README_DATA_COLLECTION.md) - Overview
- [test_free_data_migration.py](../test_free_data_migration.py) - Validation suite

### Dataset Info
- **Source**: https://github.com/philippdubach/options-data
- **License**: Educational and research purposes
- **Hosted by**: Philipp Dubach on Cloudflare R2

### Getting Help

**Issues with queries:**
- Check terminal logs for detailed errors
- Validate ticker: `has_dataset_coverage(symbol)` from `dataset_catalog`
- Ensure date range is within 2008-2025

**Issues with data quality:**
- Run schema validation: `python test_free_data_migration.py`
- Check for schema drift
- Report upstream to philippdubach GitHub

---

**Happy Data Collecting with Free, Production-Grade Options Data!**
