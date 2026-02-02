"""
Column mapping for EODHD data to standard column names.

Maps EODHD API field names to the standardized column names used
throughout the system (compatible with SimFin naming conventions).
"""

from typing import Dict

# =============================================================================
# EODHD TO STANDARD COLUMN MAPPINGS
# =============================================================================

EODHD_PRICE_MAPPING: Dict[str, str] = {
    "code": "Ticker",
    "date": "Date",
    "adjusted_close": "Adj. Close",
    "close": "Close",
    "open": "Open",
    "high": "High",
    "low": "Low",
    "volume": "Volume",
}

EODHD_INCOME_MAPPING: Dict[str, str] = {
    # Metadata
    "code": "Ticker",
    "filing_date": "Publish Date",
    "date": "Report Date",
    "fiscalYear": "Fiscal Year",
    "fiscalQuarter": "Fiscal Period",

    # Income statement items
    "totalRevenue": "Revenue",
    "costOfRevenue": "Cost of Revenue",
    "grossProfit": "Gross Profit",
    "operatingIncome": "Operating Income (Loss)",
    "interestExpense": "Interest Expense, Net",
    "incomeBeforeTax": "Pretax Income (Loss)",
    "incomeTaxExpense": "Income Tax (Expense) Benefit, Net",
    "netIncome": "Net Income",
    "netIncomeApplicableToCommonShares": "Net Income (Common)",
    "depreciation": "Depreciation & Amortization",
    "researchDevelopment": "Research & Development",

    # Alternate field names EODHD may use
    "researchAndDevelopmentExpenses": "Research & Development",
    "depreciationAndAmortization": "Depreciation & Amortization",
}

EODHD_BALANCE_MAPPING: Dict[str, str] = {
    # Metadata
    "code": "Ticker",
    "filing_date": "Publish Date",
    "date": "Report Date",
    "fiscalYear": "Fiscal Year",
    "fiscalQuarter": "Fiscal Period",

    # Balance sheet items
    "totalAssets": "Total Assets",
    "totalStockholderEquity": "Total Equity",
    "totalShareholderEquity": "Total Equity",  # Alternate spelling
    "totalCurrentAssets": "Total Current Assets",
    "totalCurrentLiabilities": "Total Current Liabilities",
    "inventory": "Inventories",
    "netReceivables": "Accounts & Notes Receivable",
    "accountsReceivable": "Accounts & Notes Receivable",  # Alternate
    "shortTermDebt": "Short Term Debt",
    "shortLongTermDebt": "Short Term Debt",  # Alternate
    "longTermDebt": "Long Term Debt",
    "longTermDebtNoncurrent": "Long Term Debt",  # Alternate
    "cashAndShortTermInvestments": "Cash, Cash Equivalents & Short Term Investments",
    "cash": "Cash, Cash Equivalents & Short Term Investments",  # Alternate
    "cashAndCashEquivalents": "Cash, Cash Equivalents & Short Term Investments",
    "propertyPlantEquipment": "Property, Plant & Equipment, Net",
    "propertyPlantAndEquipmentNet": "Property, Plant & Equipment, Net",  # Alternate
    "totalLiabilities": "Total Liabilities",
    "accountsPayable": "Payables & Accruals",
    # Shares outstanding for market cap calculation
    "commonStockSharesOutstanding": "Shares Outstanding",
    "sharesOutstanding": "Shares Outstanding",  # Alternate field name
}

EODHD_CASHFLOW_MAPPING: Dict[str, str] = {
    # Metadata
    "code": "Ticker",
    "filing_date": "Publish Date",
    "date": "Report Date",
    "fiscalYear": "Fiscal Year",
    "fiscalQuarter": "Fiscal Period",

    # Cash flow items
    "totalCashFromOperatingActivities": "Net Cash from Operating Activities",
    "operatingCashFlow": "Net Cash from Operating Activities",  # Alternate
    "capitalExpenditures": "Change in Fixed Assets & Intangibles",
    "dividendsPaid": "Dividends Paid",
    "dividendPayout": "Dividends Paid",  # Alternate
    "issuanceOfStock": "Cash from (Repurchase of) Equity",
    "repurchaseOfStock": "Cash from (Repurchase of) Equity",  # Note: sign may need adjustment
}

EODHD_COMPANY_MAPPING: Dict[str, str] = {
    "Code": "Ticker",
    "Name": "Company Name",
    "Exchange": "Exchange",
    "Type": "Security Type",
    "Sector": "Sector",
    "Industry": "Industry",
    "GicSector": "GICS Sector",
    "GicGroup": "GICS Group",
    "GicIndustry": "GICS Industry",
    "GicSubIndustry": "GICS SubIndustry",
}

# =============================================================================
# EODHD MACRO DATA MAPPING
# =============================================================================

EODHD_MACRO_MAPPING: Dict[str, str] = {
    # Economic indicators available from EODHD
    "gdp_current_usd": "GDP",
    "real_interest_rate": "REAL_INT_RATE",
    "inflation_consumer_prices_annual": "INFLATION",
    "interest_rate": "INTEREST_RATE",
    "unemployment_total": "UNEMPLOYMENT",
}

# FRED indicators we use (for reference)
FRED_INDICATORS = {
    "FEDFUNDS": "Federal Funds Rate",
    "DGS10": "10-Year Treasury Yield",
    "USACPIALLMINMEI": "Consumer Price Index",
    "GDP": "Gross Domestic Product",
}

# =============================================================================
# SIGN CONVENTIONS
# =============================================================================

# Fields that need sign adjustment (EODHD uses opposite sign from SimFin)
SIGN_FLIP_FIELDS = [
    "Income Tax (Expense) Benefit, Net",  # EODHD: positive = expense, SimFin: negative = expense
    "Change in Fixed Assets & Intangibles",  # CapEx sign conventions
]

# Fields where EODHD provides positive values but we need negative
NEGATE_FIELDS = [
    # CapEx is typically positive in EODHD but we store it as outflow (negative)
]

# =============================================================================
# SECURITY TYPE FILTERS
# =============================================================================

# Security types to include (common stocks only per Wynne thesis)
COMMON_STOCK_TYPES = [
    "Common Stock",
    "common stock",
    "COMMON_STOCK",
    "Equity",
]

# Security types to exclude
EXCLUDED_TYPES = [
    "ETF",
    "Fund",
    "Preferred",
    "ADR",
    "REIT",
    "Trust",
    "LP",
    "Warrant",
]

# =============================================================================
# FINANCIAL SECTOR IDENTIFIERS
# =============================================================================

# GICS Sector codes for financials
FINANCIAL_GICS_SECTORS = ["40", "Financials"]

# Industry keywords to identify financial companies
FINANCIAL_KEYWORDS = [
    "bank",
    "insurance",
    "financial",
    "asset management",
    "capital markets",
    "mortgage",
    "credit",
    "investment",
]

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def map_columns(df, mapping: Dict[str, str], keep_unmapped: bool = True):
    """
    Map DataFrame columns using a mapping dictionary.

    Args:
        df: DataFrame to map
        mapping: Dict of old_name -> new_name
        keep_unmapped: Whether to keep columns not in mapping

    Returns:
        DataFrame with renamed columns
    """
    import pandas as pd

    # Create reverse mapping for columns that exist
    rename_dict = {}
    for old_name, new_name in mapping.items():
        if old_name in df.columns:
            rename_dict[old_name] = new_name

    # Rename
    result = df.rename(columns=rename_dict)

    # Drop unmapped if requested
    if not keep_unmapped:
        mapped_cols = list(rename_dict.values())
        result = result[[c for c in result.columns if c in mapped_cols]]

    return result


def apply_sign_conventions(df, negate_cols=None, flip_cols=None):
    """
    Apply sign conventions to match SimFin format.

    Args:
        df: DataFrame
        negate_cols: Columns to negate (multiply by -1)
        flip_cols: Columns where positive/negative meaning is swapped

    Returns:
        DataFrame with adjusted signs
    """
    import pandas as pd

    result = df.copy()

    if negate_cols:
        for col in negate_cols:
            if col in result.columns:
                result[col] = -result[col]

    if flip_cols:
        for col in flip_cols:
            if col in result.columns:
                result[col] = -result[col]

    return result


def standardize_fiscal_period(period) -> str:
    """
    Convert various fiscal period formats to standard Q1-Q4.

    Args:
        period: Fiscal period value (1, 2, 3, 4, "Q1", "1Q", etc.)

    Returns:
        Standardized period string (Q1, Q2, Q3, Q4)
    """
    if period is None:
        return None

    period_str = str(period).upper().strip()

    # Already in Q# format
    if period_str in ["Q1", "Q2", "Q3", "Q4"]:
        return period_str

    # Numeric format
    if period_str in ["1", "2", "3", "4"]:
        return f"Q{period_str}"

    # #Q format
    if period_str in ["1Q", "2Q", "3Q", "4Q"]:
        return f"Q{period_str[0]}"

    return period_str


def is_common_stock(security_type: str) -> bool:
    """
    Check if security type indicates common stock.

    Args:
        security_type: Security type string

    Returns:
        True if common stock
    """
    if security_type is None:
        return False

    type_lower = str(security_type).lower()

    # Check for explicit exclusions first
    for excluded in EXCLUDED_TYPES:
        if excluded.lower() in type_lower:
            return False

    # Check for common stock indicators
    for common in COMMON_STOCK_TYPES:
        if common.lower() in type_lower:
            return True

    return False


def is_financial_company(sector: str = None, industry: str = None) -> bool:
    """
    Check if company is in financial sector.

    Args:
        sector: Sector name or code
        industry: Industry name

    Returns:
        True if financial company
    """
    # Check sector
    if sector:
        sector_str = str(sector).lower()
        for fin_sector in FINANCIAL_GICS_SECTORS:
            if fin_sector.lower() in sector_str:
                return True

    # Check industry keywords
    if industry:
        industry_lower = str(industry).lower()
        for keyword in FINANCIAL_KEYWORDS:
            if keyword in industry_lower:
                return True

    return False
