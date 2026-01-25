# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Machine learning-based long-term stock selection system based on Morgan Wynne's 2023 Master's thesis. Uses 50 years of North American stock fundamental data to predict 1-year forward returns using Random Forest and LSTM models.

## Key Commands

### Build the Panel Dataset
```bash
python improved_simfin_panel.py \
    --cache-dir /path/to/simfin/data \
    --out data/simfin_panel.csv \
    --add-macro \
    --exclude-financials \
    --winsorize \
    --year-adjusted-caps
```

The `--cache-dir` should contain SimFin CSV files (us-shareprices-daily.csv, us-income-quarterly.csv, etc.).

### Run Quarterly Backtest
```bash
python quarterly_rf_fixed.py
```

Other backtest variants:
- `quarterly_backtest_strict.py` - Strict point-in-time backtest
- `quarterly_backtest_walkforward.py` - Walk-forward validation
- `quarterly_backtest_arbitrary.py` - Custom date range backtest

### Run Notebooks
Jupyter notebooks follow numbered workflow: `1. Cleaning` → `2. EDA` → `3. Modelling` → `4. Evaluation`

## Architecture

### Data Pipeline (`improved_simfin_panel.py`)
Builds a monthly point-in-time (PIT) panel with proper data alignment to prevent lookahead bias:

1. `build_monthly_prices()` - Monthly OHLCV with trailing 12-month dividends
2. `build_quarterly_fundamentals()` - Merges income statement, balance sheet, cash flow
3. `merge_monthly_with_fundamentals()` - PIT-aligned merge using `merge_asof()` with backward direction
4. `apply_year_adjusted_market_caps()` - Dynamic cap classification scaled by median market cap per year
5. `add_fred_macro()` - Integrates Fed economic indicators (FEDFUNDS, DGS10, CPI, GDP)

Output: `data/simfin_panel.csv` (~360k rows × 128 columns)

### Model Class (`StockSelectionRF`)
Located in `quarterly_rf_fixed.py` and other RF scripts. Key methods:
- `prepare_features()` - Defines 7 feature categories (valuation, profitability, solvency, liquidity, efficiency, financial soundness, other + macro)
- `neutralize_features()` - Sector-based z-score normalization
- `handle_missing_data()` - Drops high-missing features then incomplete rows
- `fit()` / `predict()` - Standard sklearn interface

### Backtesting (`quarterly_backtest_*.py`)
Expanding window validation from 1981-2022. Key parameters:
- `portfolio_sizes=[30, 50, 100, 200]`
- `min_market_cap='Mid Cap'` for restricted universe
- `start_year`, `end_year` for date range

## Critical Patterns

### Point-in-Time (PIT) Compliance
All merges use 1-month lag to prevent lookahead bias:
- Fundamental data: `--fundamental-lag-months 1`
- Macro data: `--macro-shift-months 1`

### Feature Categories
```python
valuation = ['bm', 'ptb', 'ps', 'pcf', 'pe_inc', ...]
profitability = ['npm', 'gpm', 'roa', 'roe', 'roce', ...]
solvency = ['de_ratio', 'debt_at', 'intcov', ...]
liquidity = ['curr_ratio', 'quick_ratio', 'cash_ratio', ...]
efficiency = ['at_turn', 'inv_turn', 'rect_turn', ...]
financial_soundness = ['cash_lt', 'ocf_lct', 'cash_debt', ...]
macro = ['FEDFUNDS', 'DGS10', '1mo_inf_rate', '1yr_inf_rate', ...]
```

### Missing Data Strategy
- No imputation - missing data is informative
- RF: Drop features with >50% missing, then drop incomplete rows
- LSTM: Replace with -1 sentinel (outside [0,1] normalized range)

### Year-Adjusted Market Caps
Cap boundaries scale with median market cap each year (base year 2021). Categories: Nano → Micro → Small → Mid → Large → Mega Cap.

## Dependencies

pandas, numpy, scikit-learn, PyTorch, yfinance, pandas-datareader, matplotlib, seaborn

No requirements.txt exists; assumes standard Anaconda environment.

## Data Sources

- **SimFin**: Stock prices, quarterly fundamentals (70+ financial ratios)
- **FRED**: Macroeconomic indicators via pandas-datareader
- **yfinance**: S&P 500 benchmark prices for comparison
