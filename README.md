# Long-Term Stock Selection System

Production-grade stock selection and trading system based on the Wynne (2023) thesis methodology, with IBKR integration and Ireland tax tracking.

## Features

- **ML-based Stock Selection**: Random Forest model trained on 50+ years of fundamental data
- **Point-in-Time Compliance**: Strict PIT data alignment to prevent lookahead bias
- **IBKR Integration**: Paper and live trading via Interactive Brokers API
- **Ireland Tax Tracking**: CGT calculation (33% rate, €1,270 exemption), dividend withholding credits
- **Quarterly Rebalancing**: Semi-automated workflow with approval gates

## Installation

```bash
pip install -r requirements.txt
```

For IBKR integration:
```bash
pip install ib_insync
```

## Quick Start

### Run Backtest
```bash
python -m src.cli backtest --data data/simfin_panel.csv --start 2018 --end 2024 --stocks 15
```

### Paper Trading
```bash
python -m src.cli paper-trade --stocks 15 --dry-run
```

### Generate Tax Report
```bash
python -m src.cli tax-report --year 2024
```

### Check System Status
```bash
python -m src.cli status
```

## Project Structure

```
src/
├── config.py              # Central configuration
├── cli.py                 # Command-line interface
├── models/
│   ├── stock_selection_rf.py   # Random Forest model
│   └── feature_engineering.py  # Feature calculations
├── data/
│   ├── simfin_loader.py        # SimFin data loading
│   ├── panel_builder.py        # Panel construction
│   └── cache_manager.py        # Data caching
├── trading/
│   ├── ibkr_client.py          # IBKR API wrapper
│   ├── order_generator.py      # Order generation
│   ├── fee_calculator.py       # IBKR fee calculations
│   └── execution_engine.py     # Trade execution
├── tax/
│   ├── ireland_cgt.py          # Ireland CGT calculator
│   ├── dividend_tracker.py     # Dividend tracking
│   └── cost_basis.py           # FIFO cost basis
├── backtest/
│   ├── engine.py               # Backtest engine
│   ├── metrics.py              # Performance metrics
│   └── benchmark.py            # S&P 500 benchmark
└── reports/
    └── charts.py               # Visualization
```

## Configuration

User settings in `config/settings.yaml`:
- Portfolio size and rebalancing frequency
- Tax jurisdiction settings
- IBKR connection parameters

Feature definitions in `config/features.yaml`.

## Data Requirements

SimFin data files in `data/simfin/`:
- `us-shareprices-daily.csv`
- `us-income-quarterly.csv`
- `us-balance-quarterly.csv`
- `us-cashflow-quarterly.csv`
- `us-companies.csv`
- `industries.csv`

Build the panel:
```bash
python -m src.cli build-panel --add-macro
```

## Testing

```bash
pytest tests/ -v
```

## References

Based on: Wynne, M. (2023). "Long-term Stock Selection using Random Forest and LSTM Models for Fundamental Analysis"

## License

MIT
