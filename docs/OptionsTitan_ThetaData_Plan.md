# OptionsTitan — Multi-Year, Multi-Ticker Options Dataset Plan (ThetaData-first)

> Goal: Expand OptionsTitan training data from SPY + 60 days to a multi-year (2019–2024+), multi-ticker, options-focused dataset with consistent sampling across strikes/expiries and strong regime coverage.

---

## 1) Key Reality Check (Important)

### 1.1 yfinance is not sufficient for historical option chains
- yfinance is solid for multi-year underlying OHLCV.
- yfinance does not reliably provide historical option chains per day (mostly current chain; contract time-series requires knowing contract symbols, which is the hard part).

Conclusion: Use yfinance for:
- underlying OHLCV
- VIX series (^VIX)
- broad index proxies

Use ThetaData (or similar) for multi-year historical options.

---

## 2) Recommended Universe (45 tickers)

### Volume tier shorthand (estimates)
- Ultra: 200k–1M+ contracts/day
- High: 50k–200k/day
- Solid: 10k–50k/day

### Typical IV bands
- Low: 15–30%
- Mid: 25–50%
- High: 45–90%
- Extreme: 80–200%+

### ETFs
SPY, QQQ, IWM, DIA, XLF, XLK, XLE

### Mega-cap / high-liquidity tech
AAPL, MSFT, NVDA, AMZN, GOOGL, META, TSLA, AMD, NFLX, ORCL

### Financials
JPM, BAC, GS, MS, WFC, SCHW

### Healthcare / Pharma
UNH, JNJ, PFE, ABBV, LLY

### Energy
XOM, CVX, COP, SLB

### Consumer
WMT, COST, TGT, DIS, NKE

### High-volatility
GME, AMC, PLTR, COIN, ROKU, SNAP, DKNG, MARA

### Low-volatility
KO, PG, PEP, MCD, V

---

## 3) Data Collection Strategy (ThetaData-first)

### Underlying + VIX
- Pull daily OHLCV from 2019-01-01 to present.
- Pull VIX (^VIX) daily.
- Use explicit start/end dates for reproducibility.

### Historical Options — ThetaData
Pull:
- Daily option bars (EOD-style)
- Implied volatility
- Volume
- Open interest (if available)
- Bid/ask (if available)

Prefer ThetaData REST API over deprecated Python client.

---

## 4) Contract Sampling (Controls Dataset Size)

Per ticker per trading day:

Expiries:
- 25–35 DTE
- 45–75 DTE

Moneyness buckets:
- -5%, -2.5%, 0%, +2.5%, +5%

Include:
- Call and Put

Total:
2 expiries × 5 strikes × 2 types = 20 contracts/day/ticker

Liquidity filters:
- volume > 0
- open_interest >= 100 (if available)

---

## 5) Required CSV Schema Mapping

Columns:
- price — underlying price
- option_price — option premium (close or mid)
- strike_distance — strike - price
- time_to_expiry — DTE
- volume — option volume
- implied_volatility — decimal IV
- vix_level — VIX close
- spy_return_5min — use daily proxy unless intraday available
- rsi — computed from underlying
- timestamp — YYYY-MM-DD HH:MM:SS

---

## 6) Extra Raw Fields to Capture

Strongly recommended:
- bid, ask, mid, spread
- open_interest
- contract_type
- expiration_date
- strike
- option_symbol

Context features:
- underlying volume
- days_to_earnings
- days_to_dividend
- rates proxy
- sector ETF return

---

## 7) Multi-Year Normalization

Price:
- use log(price)
- use returns
- use percent moneyness

Splits:
- store raw strike + expiry
- run split sanity tests (AAPL, TSLA, NVDA)

Volume:
- log1p(volume)
- rolling median normalization

Vol:
- IV percentile/rank
- IV z-score
- IV minus realized vol

---

## 8) Training Strategy

Model:
- Universal model across tickers
- Add ticker + sector categorical features

Validation:
Walk-forward:
- Train 2019–2020 → Validate 2021
- Train 2019–2021 → Validate 2022
- Train 2019–2022 → Validate 2023
- Train 2019–2023 → Test 2024

COVID:
- include but regime-balance
- use VIX buckets
- apply sample weights

---

## 9) Storage

Canonical:
- Parquet partitioned by ticker/year

Example:
data/options/ticker=AAPL/year=2020/part.parquet

CSV:
- export only when needed

---

## 10) Sample Multi-Year Rows (Example)

timestamp,price,option_price,strike_distance,time_to_expiry,volume,implied_volatility,vix_level,spy_return_5min,rsi
2019-01-15 15:30:00,39.40,0.95,0.60,32,18450,0.19,14.2,0.0012,58.3
2020-03-16 10:00:00,57.20,12.40,-2.20,22,143200,1.10,82.7,-0.0234,22.1
2021-11-22 11:05:00,152.10,4.25,3.20,33,73400,0.28,18.6,0.0016,61.2
2022-10-13 14:25:00,140.80,7.15,-3.00,58,105600,0.46,32.6,0.0064,44.1
2024-12-13 14:55:00,205.30,3.75,-1.80,45,68910,0.26,16.8,0.0019,55.6

---

## 11) Next Step

Run one-ticker backfill (AAPL):
- 2019–2024
- 20 contracts/day sampling
- validate schema, splits, missingness

Then scale to full universe.
