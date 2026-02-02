"""
Central configuration for the stock trading system.

This module serves as the single source of truth for all constants,
hyperparameters, and configuration settings used throughout the system.
"""

from pathlib import Path
from typing import Dict, List, Any
import logging

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_LEVEL = logging.INFO


def setup_logging(level: int = LOG_LEVEL) -> None:
    """Configure logging for the application."""
    logging.basicConfig(format=LOG_FORMAT, level=level)


# =============================================================================
# PATH CONFIGURATION
# =============================================================================

# Project root (parent of src/)
PROJECT_ROOT = Path(__file__).parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
SIMFIN_DIR = DATA_DIR / "simfin"
EODHD_DIR = DATA_DIR / "eodhd"
PANELS_DIR = DATA_DIR / "panels"
CACHE_DIR = DATA_DIR / "cache"
TRADES_DIR = DATA_DIR / "trades"
PORTFOLIOS_DIR = DATA_DIR / "portfolios"

# Output directories
RESULTS_DIR = PROJECT_ROOT / "results"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = RESULTS_DIR / "figures"

# Default file paths
DEFAULT_PANEL_PATH = DATA_DIR / "simfin_panel.csv"
CACHED_PANEL_PATH = CACHE_DIR / "simfin_panel.parquet"
FRED_MACRO_CACHE = DATA_DIR / "fred_macro_cache.csv"
EODHD_MACRO_CACHE = EODHD_DIR / "macro.parquet"
BENCHMARK_CACHE = CACHE_DIR / "sp500_benchmark.parquet"

# Data source options
DATA_SOURCES = ["simfin", "eodhd"]
DEFAULT_DATA_SOURCE = "simfin"
MACRO_SOURCES = ["fred", "eodhd"]
DEFAULT_MACRO_SOURCE = "fred"

# =============================================================================
# FEATURE CATEGORIES (Single source of truth)
# =============================================================================

FEATURE_CATEGORIES: Dict[str, List[str]] = {
    "valuation": [
        "bm",           # Book-to-market
        "ptb",          # Price-to-book
        "ps",           # Price-to-sales
        "pcf",          # Price-to-cash-flow
        "pe_inc",       # P/E (income)
        "pe_exi",       # P/E (excluding items)
        "pe_op_basic",  # P/E (operating, basic)
        "pe_op_dil",    # P/E (operating, diluted)
        "evm",          # EV/EBITDA
        "dpr",          # Dividend payout ratio
        "divyield",     # Dividend yield
    ],
    "profitability": [
        "npm",          # Net profit margin
        "gpm",          # Gross profit margin
        "roa",          # Return on assets
        "roe",          # Return on equity
        "roce",         # Return on capital employed
        "opmbd",        # Operating margin before depreciation
        "opmad",        # Operating margin after depreciation
        "ptpm",         # Pre-tax profit margin
        "cfm",          # Cash flow margin
        "efftax",       # Effective tax rate
        "GProf",        # Gross profitability (GP/Assets)
        "aftret_eq",    # After-tax return on equity
        "aftret_equity",  # After-tax return on total equity
        "pretret_noa",  # Pre-tax return on net operating assets
    ],
    "solvency": [
        "de_ratio",     # Debt-to-equity
        "debt_at",      # Debt-to-assets
        "debt_assets",  # Total debt / assets
        "debt_capital", # Debt-to-capital
        "capital_ratio",  # Equity / Assets
        "intcov",       # Interest coverage (EBIT/Interest)
        "intcov_ratio", # Interest coverage ratio
        "dltt_be",      # Long-term debt / book equity
        "int_debt",     # Interest / Total debt
        "int_totdebt",  # Interest / Total debt (alt)
    ],
    "liquidity": [
        "curr_ratio",   # Current ratio
        "quick_ratio",  # Quick ratio
        "cash_ratio",   # Cash ratio
        "cash_conversion",  # OCF / Net Income
    ],
    "efficiency": [
        "at_turn",      # Asset turnover
        "inv_turn",     # Inventory turnover
        "rect_turn",    # Receivables turnover
        "pay_turn",     # Payables turnover
        "sale_invcap",  # Sales / Invested capital
        "sale_equity",  # Sales / Equity
        "sale_nwc",     # Sales / Net working capital
    ],
    "financial_soundness": [
        "cash_lt",      # Cash / Total assets
        "invt_act",     # Inventory / Current assets
        "rect_act",     # Receivables / Current assets
        "short_debt",   # Short-term debt / Total debt
        "curr_debt",    # Current debt / Total debt
        "lt_debt",      # Long-term debt / Total debt
        "profit_lct",   # Net income / Current liabilities
        "ocf_lct",      # OCF / Current liabilities
        "cash_debt",    # Cash / Total debt
        "fcf_ocf",      # FCF / OCF
    ],
    "other": [
        "accrual",      # Accruals / Assets
        "rd_sale",      # R&D / Sales
        "lt_ppent",     # PP&E / Assets
    ],
    "macro": [
        "FEDFUNDS",     # Federal Funds Rate
        "DGS10",        # 10-Year Treasury
        "USACPIALLMINMEI",  # CPI
        "1mo_inf_rate", # 1-month inflation rate
        "1yr_inf_rate", # 1-year inflation rate
        "GDP",          # GDP level
        "1mo_GDP",      # 1-month GDP change
        "1yr_GDP",      # 1-year GDP change
    ],
    "size": [
        "MthCap",       # Market capitalization
    ],
}

# All features flattened
ALL_FEATURES: List[str] = [
    feat for category in FEATURE_CATEGORIES.values() for feat in category
]

# Features that should NOT be sector-neutralized (macro indicators)
NON_NEUTRALIZED_FEATURES: List[str] = (
    FEATURE_CATEGORIES["macro"] + FEATURE_CATEGORIES["size"]
)

# =============================================================================
# RANDOM FOREST HYPERPARAMETERS
# =============================================================================

RF_PARAMS: Dict[str, Any] = {
    "n_estimators": 500,
    "max_depth": 12,
    "max_features": 0.3,
    "min_samples_leaf": 50,
    "max_samples": 0.7,
    "random_state": 42,
    "n_jobs": -1,
}

# Training parameters
MISSING_THRESHOLD_PCT = 0.5  # Drop features with >50% missing
ROW_COMPLETENESS_THRESHOLD = 0.8  # Drop rows with <80% complete features
WINSORIZE_PERCENTILES = (1, 99)  # Winsorize targets at 1st and 99th percentile

# =============================================================================
# MARKET CAP BOUNDARIES (2021 Base Year)
# =============================================================================

MARKET_CAP_BASE_YEAR = 2021

# Base boundaries from Investopedia/Wynne (2023)
MARKET_CAP_BOUNDARIES: Dict[str, float] = {
    "Nano Cap": 50e6,
    "Micro Cap": 300e6,
    "Small Cap": 2e9,
    "Mid Cap": 10e9,
    "Large Cap": 200e9,
    # Mega Cap = above Large Cap
}

# Ordered hierarchy for filtering
MARKET_CAP_HIERARCHY: List[str] = [
    "Nano Cap",
    "Micro Cap",
    "Small Cap",
    "Mid Cap",
    "Large Cap",
    "Mega Cap",
]

# =============================================================================
# IBKR FEE STRUCTURE
# =============================================================================

IBKR_FEES: Dict[str, Dict[str, Any]] = {
    "tiered": {
        "us_stocks": {
            "per_share": 0.0035,
            "min_per_order": 0.35,
            "max_pct": 0.01,
        },
        "exchange_fees": {
            "remove_liquidity": 0.003,  # Per share, approximate
            "add_liquidity": -0.002,    # Rebate per share, approximate
        },
    },
    "fixed": {
        "us_stocks": {
            "per_share": 0.005,
            "min_per_order": 1.00,
            "max_pct": 0.01,
        },
    },
}

# Default pricing model for small accounts
DEFAULT_IBKR_PRICING = "tiered"

# IBKR connection ports
IBKR_PORTS: Dict[str, int] = {
    "paper": 7497,
    "live": 7496,
}

# =============================================================================
# IRELAND TAX CONFIGURATION
# =============================================================================

IRELAND_TAX: Dict[str, Any] = {
    # Capital Gains Tax
    "cgt_rate": 0.33,
    "annual_exemption": 1270,  # EUR

    # US dividend withholding (with W-8BEN)
    "us_withholding_rate": 0.15,

    # Without W-8BEN
    "us_withholding_rate_no_treaty": 0.30,

    # Dividend income is taxed as regular income in Ireland
    # (actual rate depends on individual's marginal rate)
    "dividend_prsi_rate": 0.04,  # PRSI on investment income
    "dividend_usc_rate": 0.08,   # USC on investment income (approx)
}

# Cost basis method
COST_BASIS_METHOD = "FIFO"  # First In, First Out

# =============================================================================
# BACKTEST CONFIGURATION
# =============================================================================

BACKTEST_DEFAULTS: Dict[str, Any] = {
    "rebalance_freq": "Q",      # Quarterly
    "n_stocks": 15,             # Number of stocks in portfolio
    "initial_capital": 30000,   # EUR
    "min_market_cap": "Mid Cap",
    "start_year": 2017,
    "end_year": 2024,
}

# Map rebalancing frequency to appropriate training target
# Train on returns matching the holding period
REBALANCE_TO_TARGET: Dict[str, str] = {
    "M": "1mo_return",    # Monthly rebalancing → 1-month forward return
    "Q": "3mo_return",    # Quarterly rebalancing → 3-month forward return
    "S": "6mo_return",    # Semi-annual rebalancing → 6-month forward return
    "A": "1yr_return",    # Annual rebalancing → 1-year forward return
    "Y": "1yr_return",    # Alias for annual
}

# Portfolio size options
PORTFOLIO_SIZES: List[int] = [10, 15, 20, 30, 50]

# =============================================================================
# DATA PIPELINE CONFIGURATION
# =============================================================================

DATA_PIPELINE: Dict[str, Any] = {
    "fundamental_lag_months": 1,  # PIT lag for fundamentals
    "macro_shift_months": 1,      # PIT lag for macro data
    "winsorize_mad": 5.0,         # MAD multiplier for winsorization
    "exclude_financials": True,
    "apply_quality_filters": True,
    "year_adjusted_caps": True,
}

# Quality filter thresholds
QUALITY_FILTERS: Dict[str, Any] = {
    "min_price": 1.0,           # Exclude penny stocks
    "min_market_cap": 5e6,      # Minimum market cap
    "min_book_equity": 0,       # Positive book equity required
    "return_bounds": (-0.99, 10.0),  # Valid return range
}

# =============================================================================
# SIMFIN DATA COLUMNS
# =============================================================================

SIMFIN_PRICE_COLS = [
    "Ticker",
    "SimFinId",
    "Date",
    "Adj. Close",
    "Dividend",
    "Shares Outstanding",
]

SIMFIN_INCOME_COLS = [
    "Ticker",
    "Fiscal Year",
    "Fiscal Period",
    "Publish Date",
    "Revenue",
    "Cost of Revenue",
    "Gross Profit",
    "Operating Income (Loss)",
    "Interest Expense, Net",
    "Pretax Income (Loss)",
    "Income Tax (Expense) Benefit, Net",
    "Net Income",
    "Net Income (Common)",
    "Depreciation & Amortization",
    "Research & Development",
]

SIMFIN_BALANCE_COLS = [
    "Ticker",
    "Fiscal Year",
    "Fiscal Period",
    "Publish Date",
    "Total Assets",
    "Total Equity",
    "Total Current Assets",
    "Total Current Liabilities",
    "Inventories",
    "Cash, Cash Equivalents & Short Term Investments",
    "Accounts & Notes Receivable",
    "Payables & Accruals",
    "Property, Plant & Equipment, Net",
    "Short Term Debt",
    "Long Term Debt",
    "Total Liabilities",
]

SIMFIN_CASHFLOW_COLS = [
    "Ticker",
    "Fiscal Year",
    "Fiscal Period",
    "Publish Date",
    "Net Cash from Operating Activities",
    "Change in Fixed Assets & Intangibles",
    "Dividends Paid",
    "Cash from (Repurchase of) Equity",
]

# =============================================================================
# EODHD CONFIGURATION
# =============================================================================

EODHD_CONFIG: Dict[str, Any] = {
    "base_url": "https://eodhd.com/api",
    "exchange": "US",
    "bulk_endpoint": "eod-bulk-last-day",
    "fundamentals_endpoint": "fundamentals",
    "exchange_symbols_endpoint": "exchange-symbol-list",
    # API rate limits
    "requests_per_minute": 1000,
    "bulk_batch_size": 500,
    # Data filters
    "securities": "common-stock",  # Options: common-stock, all
    "min_market_cap": 5e6,
    # Historical depth
    "start_date": "1995-01-01",
}

# EODHD file names in cache directory
EODHD_FILES = {
    "prices": "prices.parquet",
    "income": "income.parquet",
    "balance": "balance.parquet",
    "cashflow": "cashflow.parquet",
    "companies": "companies.parquet",
    "companies_detail": "companies_detail.parquet",  # Sector/industry from fundamentals API
    "fundamentals_all": "fundamentals_all.parquet",  # Complete fundamentals (all columns)
    "macro": "macro.parquet",
    "metadata": "metadata.json",
}

# EODHD columns to request for fundamentals
EODHD_FUNDAMENTAL_FIELDS = [
    # Income statement
    "totalRevenue",
    "costOfRevenue",
    "grossProfit",
    "operatingIncome",
    "interestExpense",
    "incomeBeforeTax",
    "incomeTaxExpense",
    "netIncome",
    "netIncomeApplicableToCommonShares",
    "depreciation",
    "researchDevelopment",
    # Balance sheet
    "totalAssets",
    "totalStockholderEquity",
    "totalCurrentAssets",
    "totalCurrentLiabilities",
    "inventory",
    "netReceivables",
    "shortTermDebt",
    "longTermDebt",
    "cashAndShortTermInvestments",
    "propertyPlantEquipment",
    "totalLiabilities",
    "accountsPayable",
    # Cash flow
    "totalCashFromOperatingActivities",
    "capitalExpenditures",
    "dividendsPaid",
]

# EODHD macro indicators
EODHD_MACRO_INDICATORS = [
    "gdp_current_usd",
    "real_interest_rate",
    "inflation_consumer_prices_annual",
    "interest_rate",
    "unemployment_total",
]

# =============================================================================
# EPSILON FOR NUMERICAL STABILITY
# =============================================================================

EPS = 1e-12
