# CLAUDE.md

Project guidance for Claude Code when working with this repository.

## Project Overview

ML-based long-term stock selection system using Random Forest, based on Wynne (2023) thesis. Uses SimFin fundamental data to predict 1-year forward returns.

## Key Commands

### Build Panel Dataset
```bash
python -m src.cli build-panel --simfin-dir data/simfin --output data/simfin_panel.csv
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
│   ├── panel_builder.py        # Builds panel from SimFin CSVs
│   └── simfin_loader.py        # Loads SimFin files
├── backtest/
│   ├── engine.py               # PIT-compliant backtest engine
│   ├── metrics.py              # Performance metrics
│   └── benchmark.py            # Benchmark data
├── trading/
│   └── fee_calculator.py       # IBKR fee calculation
└── tax/
    └── ireland_cgt.py          # Ireland CGT calculator

data/simfin/                    # SimFin source files (not in git)
config/                         # YAML configuration
tests/                          # Unit tests
docs/                           # Thesis PDF
```

## Data Pipeline

1. **SimFin CSVs** → `simfin_loader.py` loads files
2. **Daily prices** → Monthly with TTM dividends, 1yr returns
3. **Quarterly fundamentals** → TTM for flow items
4. **PIT merge** → `merge_asof` with 1-month lag
5. **70+ ratios** → Valuation, profitability, solvency, etc.
6. **Quality filters** → Min price, market cap, positive equity
7. **Year-adjusted caps** → Dynamic market cap buckets

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

## Dependencies

pandas, numpy, scikit-learn, matplotlib, yfinance, click
