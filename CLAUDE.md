# CLAUDE.md

Project guidance for Claude Code when working with this repository.

## Project Overview

ML-based long-term stock selection system using Random Forest, based on Wynne (2023) thesis. Supports multiple data sources (SimFin, EODHD) to predict 1-year forward returns.

## Key Commands

### Build Panel Dataset
```bash
# SimFin (default)
python -m src.cli build-panel --simfin-dir data/simfin --output data/simfin_panel.csv

# EODHD (requires download-eodhd first)
python -m src.cli build-panel --data-source eodhd --output data/panels/eodhd_panel.parquet --format parquet
```

### Download EODHD Data
```bash
# Set API key (or use --api-key)
export EODHD_API_KEY=your_key

# Download data (common stocks only, saves as Parquet)
python -m src.cli download-eodhd --output-dir data/eodhd

# Include macro indicators
python -m src.cli download-eodhd --include-macro
```

### Compare Data Sources
```bash
python -m src.cli compare-sources --simfin-dir data/simfin --eodhd-dir data/eodhd
```

### Run Backtest
```bash
# Annual rebalancing in March (recommended for 1-year returns)
python -m src.cli backtest --start 2016 --end 2024 --rebalance-freq A --rebalance-month 3

# Quarterly rebalancing
python -m src.cli backtest --rebalance-freq Q

# With output file
python -m src.cli backtest -o results/backtest.csv
```

### Run Tests
```bash
pytest tests/ -v
```

## Directory Structure

```
src/
├── cli.py                 # Command-line interface
├── config.py              # Central configuration
├── models/
│   └── stock_selection_rf.py   # Random Forest model
├── data/
│   ├── base_loader.py          # Abstract DataLoader interface
│   ├── simfin_loader.py        # Loads SimFin CSV files
│   ├── eodhd_loader.py         # Loads EODHD Parquet cache
│   ├── eodhd_downloader.py     # Downloads from EODHD API
│   ├── column_mapping.py       # EODHD→Standard column mappings
│   ├── panel_builder.py        # Builds panel from any data source
│   └── data_validator.py       # Validates data, compares sources
├── backtest/
│   ├── engine.py               # PIT-compliant backtest engine
│   ├── metrics.py              # Performance metrics
│   └── benchmark.py            # Benchmark data
├── trading/
│   └── fee_calculator.py       # IBKR fee calculation
└── tax/
    └── ireland_cgt.py          # Ireland CGT calculator

data/
├── simfin/                     # SimFin source files (CSV, not in git)
├── eodhd/                      # EODHD cache files (Parquet, not in git)
└── panels/                     # Built panel datasets
config/                         # YAML configuration
tests/                          # Unit tests
docs/                           # Thesis PDF
```

## Data Pipeline

Supports two data sources via `DataLoader` interface:
- **SimFin** (default): CSV files, free with registration
- **EODHD**: Parquet cache from API, requires paid subscription

Pipeline steps:
1. **Data source** → `simfin_loader.py` or `eodhd_loader.py`
2. **Daily prices** → Monthly with TTM dividends, 1yr returns
3. **Quarterly fundamentals** → TTM for flow items
4. **PIT merge** → `merge_asof` with 1-month lag
5. **70+ ratios** → Valuation, profitability, solvency, etc.
6. **Quality filters** → Min price, market cap, positive equity
7. **Year-adjusted caps** → Dynamic market cap buckets

### Data Source Options
| Source | Format | Pros | Cons |
|--------|--------|------|------|
| SimFin | CSV | Free, large history | Manual download |
| EODHD | Parquet | API access, faster | Paid, 100K calls/mo |

## Key Patterns

### Point-in-Time (PIT) Compliance
All merges use backward direction with 1-month lag to prevent lookahead bias.

### Feature Categories
- valuation: bm, ptb, ps, pcf, pe_*
- profitability: npm, gpm, roa, roe, roce
- solvency: de_ratio, debt_at, intcov
- liquidity: curr_ratio, quick_ratio, cash_ratio
- efficiency: at_turn, inv_turn, rect_turn
- macro: FEDFUNDS, DGS10, CPI, GDP

### Rebalancing Options
- Annual (A): Single rebalance per year, specify month (default: March)
- Quarterly (Q): End of Mar, Jun, Sep, Dec
- Monthly (M): Every month-end

### Benchmark Options
- `simfin`: Cap-weighted average of universe
- `yfinance`: S&P 500 index

## SimFin Source Files

Required in `data/simfin/`:
- `us-shareprices-daily.csv` (~800MB)
- `us-income-quarterly.csv`
- `us-balance-quarterly.csv`
- `us-cashflow-quarterly.csv`
- `us-companies.csv`
- `industries.csv`

## EODHD Cache Files

Created by `download-eodhd` in `data/eodhd/`:
- `prices.parquet` - Historical daily prices
- `income.parquet` - Income statement data
- `balance.parquet` - Balance sheet data
- `cashflow.parquet` - Cash flow statement data
- `companies.parquet` - Company metadata
- `macro.parquet` - Macro indicators (optional)
- `metadata.json` - Download timestamp and stats

## Dependencies

pandas, numpy, scikit-learn, matplotlib, yfinance, click, requests, pyarrow
