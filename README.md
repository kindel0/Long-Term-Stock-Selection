# Long-Term Stock Selection System

ML-based stock selection system using Random Forest, based on Wynne (2023) thesis methodology.

## Features

- **Random Forest Model**: Trained on 50+ fundamental ratios
- **Point-in-Time Compliance**: Strict PIT alignment prevents lookahead bias
- **Flexible Rebalancing**: Annual, quarterly, or monthly
- **Multiple Data Sources**: SimFin (default) or EODHD
- **Parquet Support**: Fast, compressed storage for large datasets
- **Ireland Tax Tracking**: CGT calculation (33% rate, €1,270 exemption)
- **IBKR Integration**: Live and paper trading with Interactive Brokers

## Installation

```bash
pip install -r requirements.txt
```

## Data Sources

The system supports two data providers:

| Feature | SimFin | EODHD |
|---------|--------|-------|
| Default | Yes | No |
| Format | CSV | Parquet |
| Cost | Free (with limits) | $60-80/month |
| History | ~10 years | 30+ years |
| API Limits | N/A | 100K calls/day |

### SimFin (Default)

Download SimFin bulk data files to `data/simfin/`:
- `us-shareprices-daily.csv`
- `us-income-quarterly.csv`
- `us-balance-quarterly.csv`
- `us-cashflow-quarterly.csv`
- `us-companies.csv`
- `industries.csv`

### EODHD (Alternative)

EODHD provides longer history and includes sector/industry data for filtering.

**API Subscription Required:**
- Fundamentals Data Feed (~$60/month) - for fundamental data
- EOD Historical Data (~$20/month) - for price data

**API Call Costs:**
- Fundamental requests: **10 API calls each**
- Price requests: **1 API call each**
- Daily limit: **100,000 API calls**

**Download EODHD Data:**
```bash
# Set API key
export EODHD_API_KEY=your_key

# Download all data (requires ~200K API calls = 2 days)
python -m src.cli download-eodhd

# Day 1: Will stop at limit, saves progress
# Day 2: Resume after midnight GMT
python -m src.cli download-eodhd --resume

# Download only prices (if you have fundamentals)
python -m src.cli download-eodhd --no-fundamentals

# Download only fundamentals (if you have prices)
python -m src.cli download-eodhd --no-prices
```

**EODHD Files Created:**
```
data/eodhd/
├── prices.parquet           # 48M+ price records
├── fundamentals_all.parquet # Complete fundamentals (all columns)
├── income.parquet           # Income statement data
├── balance.parquet          # Balance sheet data
├── cashflow.parquet         # Cash flow data
├── companies.parquet        # Basic company list
├── companies_detail.parquet # Sector/industry from fundamentals API
└── metadata.json            # Download stats
```

## Quick Start

### 1. Build Panel Dataset

**With SimFin (default):**
```bash
python -m src.cli build-panel --simfin-dir data/simfin --output data/simfin_panel.csv
```

**With EODHD:**
```bash
python -m src.cli build-panel --data-source eodhd --output data/eodhd_panel.parquet --format parquet
```

**Options:**
| Option | Default | Description |
|--------|---------|-------------|
| `--data-source` | simfin | Data provider: simfin or eodhd |
| `--macro-source` | fred | Macro data: fred or eodhd |
| `--format` | csv | Output format: csv or parquet |
| `--no-macro` | false | Skip macro features |
| `--no-filters` | false | Skip quality filters |

### 2. Run Backtest

```bash
# Annual rebalancing (recommended for 1-year returns)
python -m src.cli backtest --start 2016 --end 2024 --rebalance-freq A --rebalance-month 3 --stocks 15

# With output file
python -m src.cli backtest --start 2016 --end 2024 -o results/backtest.csv

# Rebuild panel before backtest
python -m src.cli backtest --rebuild-panel --start 2016 --end 2024

# Using EODHD panel
python -m src.cli backtest --data data/eodhd_panel.parquet --start 2010 --end 2024
```

### 3. Paper Trading

```bash
# Paper trade with current model
python -m src.cli paper-trade --stocks 15 --capital 10000

# With fractional shares
python -m src.cli paper-trade --stocks 15 --capital 5000 --fractional
```

### 4. Compare Data Sources

```bash
# Compare SimFin vs EODHD data quality
python -m src.cli compare-sources --output comparison_report.json
```

### 5. Other Commands

```bash
# Check system status
python -m src.cli status

# Validate panel data
python -m src.cli validate-data

# Generate tax report
python -m src.cli tax-report --year 2024

# Show current orders
python -m src.cli show-orders

# Cancel all orders
python -m src.cli cancel-orders
```

## Project Structure

```
├── src/
│   ├── cli.py                      # Command-line interface
│   ├── config.py                   # Central configuration
│   ├── models/
│   │   └── stock_selection_rf.py   # Random Forest model
│   ├── data/
│   │   ├── base_loader.py          # Abstract data loader interface
│   │   ├── simfin_loader.py        # SimFin data loading
│   │   ├── eodhd_loader.py         # EODHD data loading
│   │   ├── eodhd_downloader.py     # EODHD API downloader
│   │   ├── column_mapping.py       # EODHD to standard column mapping
│   │   ├── panel_builder.py        # Panel construction
│   │   └── data_validator.py       # Data quality validation
│   ├── backtest/
│   │   ├── engine.py               # Backtest engine
│   │   └── metrics.py              # Performance metrics
│   ├── trading/
│   │   ├── fee_calculator.py       # IBKR fee calculations
│   │   └── ibkr_integration.py     # IBKR API integration
│   └── tax/
│       └── ireland_cgt.py          # Ireland CGT calculator
├── config/
│   ├── settings.yaml               # User settings
│   └── features.yaml               # Feature definitions
├── data/
│   ├── simfin/                     # SimFin source CSVs
│   └── eodhd/                      # EODHD Parquet cache
├── tests/                          # Unit tests
└── docs/                           # Documentation
```

## CLI Reference

### backtest

| Option | Default | Description |
|--------|---------|-------------|
| `--data` | data/simfin_panel.csv | Panel dataset path |
| `--rebuild-panel` | false | Rebuild panel before backtest |
| `--start` | 2018 | Start year |
| `--end` | 2024 | End year |
| `--rebalance-freq` | A | A=annual, Q=quarterly, M=monthly |
| `--rebalance-month` | 3 | Month for annual rebalancing (1-12) |
| `--stocks` | 15 | Portfolio size |
| `--min-cap` | Mid Cap | Minimum market cap filter |
| `--benchmark-source` | simfin | simfin or yfinance |
| `--output` | - | Save results to CSV |

### download-eodhd

| Option | Default | Description |
|--------|---------|-------------|
| `--api-key` | $EODHD_API_KEY | EODHD API key |
| `--output-dir` | data/eodhd | Output directory |
| `--no-prices` | false | Skip price download |
| `--no-fundamentals` | false | Skip fundamentals download |
| `--include-macro` | false | Include macro indicators |
| `--resume` | true | Resume interrupted download |
| `--fresh` | false | Start fresh, ignore progress |

## Testing

```bash
pytest tests/ -v
```

## References

Based on: Wynne, M. (2023). "Long-term Stock Selection using Random Forest and LSTM Models for Fundamental Analysis"

## License

MIT
