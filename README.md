# Long-Term Stock Selection System

ML-based stock selection system using Random Forest, based on Wynne (2023) thesis methodology.

## Features

- **Random Forest Model**: Trained on 50+ fundamental ratios
- **Point-in-Time Compliance**: Strict PIT alignment prevents lookahead bias
- **Flexible Rebalancing**: Annual, quarterly, or monthly
- **SimFin Data**: Builds panel from SimFin fundamental data
- **Ireland Tax Tracking**: CGT calculation (33% rate, €1,270 exemption)

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Build Panel Dataset

```bash
python -m src.cli build-panel --simfin-dir data/simfin --output data/simfin_panel.csv
```

Required SimFin files in `data/simfin/`:
- `us-shareprices-daily.csv`
- `us-income-quarterly.csv`
- `us-balance-quarterly.csv`
- `us-cashflow-quarterly.csv`
- `us-companies.csv`
- `industries.csv`

### 2. Run Backtest

```bash
# Annual rebalancing (recommended for 1-year returns)
python -m src.cli backtest --start 2016 --end 2024 --rebalance-freq A --rebalance-month 3 --stocks 15

# With output file
python -m src.cli backtest --start 2016 --end 2024 -o results/backtest.csv

# Rebuild panel before backtest
python -m src.cli backtest --rebuild-panel --start 2016 --end 2024
```

### 3. Other Commands

```bash
# Check system status
python -m src.cli status

# Validate panel data
python -m src.cli validate-data

# Generate tax report
python -m src.cli tax-report --year 2024
```

## Project Structure

```
├── src/
│   ├── cli.py                 # Command-line interface
│   ├── config.py              # Central configuration
│   ├── models/
│   │   └── stock_selection_rf.py   # Random Forest model
│   ├── data/
│   │   ├── panel_builder.py        # Panel construction
│   │   └── simfin_loader.py        # SimFin data loading
│   ├── backtest/
│   │   ├── engine.py               # Backtest engine
│   │   └── metrics.py              # Performance metrics
│   ├── trading/
│   │   └── fee_calculator.py       # IBKR fee calculations
│   └── tax/
│       └── ireland_cgt.py          # Ireland CGT calculator
├── config/
│   ├── settings.yaml          # User settings
│   └── features.yaml          # Feature definitions
├── data/
│   └── simfin/                # SimFin source CSVs
├── tests/                     # Unit tests
└── docs/                      # Documentation
```

## CLI Options

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

## Testing

```bash
pytest tests/ -v
```

## References

Based on: Wynne, M. (2023). "Long-term Stock Selection using Random Forest and LSTM Models for Fundamental Analysis"

## License

MIT
