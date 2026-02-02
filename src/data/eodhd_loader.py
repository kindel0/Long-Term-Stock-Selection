"""
EODHD data loader.

Loads EODHD data from Parquet cache files and maps columns to
standard names compatible with the rest of the system.
"""

import logging
from pathlib import Path
from typing import Optional, Set

import pandas as pd

from ..config import EODHD_DIR, EODHD_FILES
from .base_loader import DataLoader
from .column_mapping import (
    EODHD_PRICE_MAPPING,
    EODHD_INCOME_MAPPING,
    EODHD_BALANCE_MAPPING,
    EODHD_CASHFLOW_MAPPING,
    EODHD_COMPANY_MAPPING,
    map_columns,
    standardize_fiscal_period,
    is_financial_company,
)

logger = logging.getLogger(__name__)


class EODHDLoader(DataLoader):
    """
    Loader for EODHD data from Parquet cache.

    Implements the DataLoader interface to provide EODHD data
    in the same format as SimFin data.
    """

    def __init__(self, data_dir: Optional[Path] = None):
        """
        Initialize the loader.

        Args:
            data_dir: Directory containing EODHD Parquet files
        """
        super().__init__(data_dir)
        self.data_dir = Path(data_dir) if data_dir else EODHD_DIR
        self._companies_cache = None

    def _check_file_exists(self, filename: str) -> Path:
        """Check if a data file exists and return its path."""
        filepath = self.data_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(
                f"EODHD file not found: {filepath}. "
                "Run 'python -m src.cli download-eodhd' to download data."
            )
        return filepath

    def _to_numeric(self, series: pd.Series) -> pd.Series:
        """Convert a series to numeric, coercing errors to NaN."""
        return pd.to_numeric(series, errors="coerce")

    def _add_fiscal_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Derive Fiscal Year and Fiscal Period from date column if not present.

        EODHD uses the period end date, so we derive:
        - Fiscal Year: year of the date
        - Fiscal Period: quarter based on month (Q1=Jan-Mar, Q2=Apr-Jun, etc.)
        """
        # Find the date column
        date_col = None
        for col in ["Report Date", "date", "Date"]:
            if col in df.columns:
                date_col = col
                break

        if date_col is None:
            return df

        # Parse date
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

        # Derive Fiscal Year if not present
        if "Fiscal Year" not in df.columns:
            df["Fiscal Year"] = df[date_col].dt.year

        # Derive Fiscal Period if not present
        if "Fiscal Period" not in df.columns:
            # Map month to quarter
            df["Fiscal Period"] = df[date_col].dt.month.map(
                lambda m: f"Q{(m - 1) // 3 + 1}" if pd.notna(m) else None
            )

        return df

    def load_prices(self) -> pd.DataFrame:
        """
        Load share price data.

        Returns:
            DataFrame with columns: Ticker, SimFinId, Date, Adj. Close,
            Dividend, Shares Outstanding
        """
        filepath = self._check_file_exists(EODHD_FILES["prices"])
        df = pd.read_parquet(filepath)

        # Map columns to standard names
        df = map_columns(df, EODHD_PRICE_MAPPING)

        # Parse dates
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Ticker", "Date"])

        # Convert numerics
        if "Adj. Close" in df.columns:
            df["Adj. Close"] = self._to_numeric(df["Adj. Close"])
        elif "Close" in df.columns:
            df["Adj. Close"] = self._to_numeric(df["Close"])

        # EODHD doesn't provide dividends in price data by default
        # Set to 0 if not present
        if "Dividend" not in df.columns:
            df["Dividend"] = 0.0
        else:
            df["Dividend"] = self._to_numeric(df["Dividend"]).fillna(0.0)

        # Shares outstanding may not be in price data
        if "Shares Outstanding" not in df.columns:
            df["Shares Outstanding"] = None
        else:
            df["Shares Outstanding"] = self._to_numeric(df["Shares Outstanding"])

        # Create SimFinId equivalent (use ticker as unique ID)
        if "SimFinId" not in df.columns:
            df["SimFinId"] = df["Ticker"]

        df = df.sort_values(["Ticker", "Date"])

        logger.info(f"Loaded {len(df)} EODHD price records for {df['Ticker'].nunique()} tickers")
        return df

    def load_income(self) -> pd.DataFrame:
        """
        Load income statement data.

        Returns:
            DataFrame with income statement line items
        """
        filepath = self._check_file_exists(EODHD_FILES["income"])
        df = pd.read_parquet(filepath)

        # Map columns
        df = map_columns(df, EODHD_INCOME_MAPPING)

        # Handle publish date (filing_date in EODHD)
        if "Publish Date" in df.columns:
            df["Publish Date"] = pd.to_datetime(df["Publish Date"], errors="coerce")
        elif "Report Date" in df.columns:
            # Fall back to report date if filing date not available
            df["Publish Date"] = pd.to_datetime(df["Report Date"], errors="coerce")

        # Derive Fiscal Year and Fiscal Period from date if not present
        df = self._add_fiscal_columns(df)

        df = df.dropna(subset=["Ticker", "Publish Date"])

        # Standardize fiscal period
        if "Fiscal Period" in df.columns:
            df["Fiscal Period"] = df["Fiscal Period"].apply(standardize_fiscal_period)

        # Convert numeric columns
        for col in df.columns:
            if col not in ["Ticker", "Fiscal Year", "Fiscal Period", "Publish Date", "Report Date"]:
                df[col] = self._to_numeric(df[col])

        # Handle sign conventions for tax expense
        # EODHD: positive = expense, SimFin: negative = expense
        if "Income Tax (Expense) Benefit, Net" in df.columns:
            df["Income Tax (Expense) Benefit, Net"] = -df["Income Tax (Expense) Benefit, Net"]

        logger.info(f"Loaded {len(df)} EODHD income records")
        return df

    def load_balance(self) -> pd.DataFrame:
        """
        Load balance sheet data.

        Returns:
            DataFrame with balance sheet line items
        """
        filepath = self._check_file_exists(EODHD_FILES["balance"])
        df = pd.read_parquet(filepath)

        # Map columns
        df = map_columns(df, EODHD_BALANCE_MAPPING)

        # Handle publish date
        if "Publish Date" in df.columns:
            df["Publish Date"] = pd.to_datetime(df["Publish Date"], errors="coerce")
        elif "Report Date" in df.columns:
            df["Publish Date"] = pd.to_datetime(df["Report Date"], errors="coerce")

        # Derive Fiscal Year and Fiscal Period from date if not present
        df = self._add_fiscal_columns(df)

        df = df.dropna(subset=["Ticker", "Publish Date"])

        # Standardize fiscal period
        if "Fiscal Period" in df.columns:
            df["Fiscal Period"] = df["Fiscal Period"].apply(standardize_fiscal_period)

        # Convert numeric columns
        for col in df.columns:
            if col not in ["Ticker", "Fiscal Year", "Fiscal Period", "Publish Date", "Report Date"]:
                df[col] = self._to_numeric(df[col])

        logger.info(f"Loaded {len(df)} EODHD balance sheet records")
        return df

    def load_cashflow(self) -> pd.DataFrame:
        """
        Load cash flow statement data.

        Returns:
            DataFrame with cash flow line items
        """
        filepath = self._check_file_exists(EODHD_FILES["cashflow"])
        df = pd.read_parquet(filepath)

        # Map columns
        df = map_columns(df, EODHD_CASHFLOW_MAPPING)

        # Handle publish date
        if "Publish Date" in df.columns:
            df["Publish Date"] = pd.to_datetime(df["Publish Date"], errors="coerce")
        elif "Report Date" in df.columns:
            df["Publish Date"] = pd.to_datetime(df["Report Date"], errors="coerce")

        # Derive Fiscal Year and Fiscal Period from date if not present
        df = self._add_fiscal_columns(df)

        df = df.dropna(subset=["Ticker", "Publish Date"])

        # Standardize fiscal period
        if "Fiscal Period" in df.columns:
            df["Fiscal Period"] = df["Fiscal Period"].apply(standardize_fiscal_period)

        # Convert numeric columns
        for col in df.columns:
            if col not in ["Ticker", "Fiscal Year", "Fiscal Period", "Publish Date", "Report Date"]:
                df[col] = self._to_numeric(df[col])

        # Handle CapEx sign convention
        # EODHD: positive = spending, we want negative (cash outflow)
        if "Change in Fixed Assets & Intangibles" in df.columns:
            # Ensure CapEx is negative (outflow)
            capex = df["Change in Fixed Assets & Intangibles"]
            df["Change in Fixed Assets & Intangibles"] = -capex.abs()

        logger.info(f"Loaded {len(df)} EODHD cash flow records")
        return df

    def load_companies(self) -> pd.DataFrame:
        """
        Load company metadata.

        Returns:
            DataFrame with company info including Industry
        """
        filepath = self.data_dir / EODHD_FILES["companies"]
        if not filepath.exists():
            logger.warning(f"EODHD companies file not found: {filepath}")
            return pd.DataFrame()

        df = pd.read_parquet(filepath)
        df = map_columns(df, EODHD_COMPANY_MAPPING)

        # Ensure we have a Ticker column (may be "Code" in raw data)
        if "Ticker" not in df.columns and "Code" in df.columns:
            df["Ticker"] = df["Code"]

        # Cache for sector lookups
        self._companies_cache = df

        logger.info(f"Loaded {len(df)} EODHD company records")
        return df

    def load_industries(self) -> pd.DataFrame:
        """
        Load industry classification data.

        For EODHD, industry info is in the companies file.

        Returns:
            DataFrame with industry and sector mappings
        """
        companies = self.load_companies()
        if companies.empty:
            return pd.DataFrame()

        # Extract unique industry/sector combinations
        cols = []
        if "Sector" in companies.columns:
            cols.append("Sector")
        if "Industry" in companies.columns:
            cols.append("Industry")
        if "GICS Sector" in companies.columns:
            cols.append("GICS Sector")

        if not cols:
            return pd.DataFrame()

        return companies[cols].drop_duplicates()

    def load_fundamentals(self) -> pd.DataFrame:
        """
        Load and merge all fundamental data (income, balance, cashflow).

        Returns:
            Merged DataFrame with all fundamental data
        """
        income = self.load_income()
        balance = self.load_balance()
        cashflow = self.load_cashflow()

        # Determine merge keys
        merge_keys = ["Ticker", "Publish Date"]

        # Add fiscal keys if available
        if "Fiscal Year" in income.columns and "Fiscal Period" in income.columns:
            merge_keys = ["Ticker", "Fiscal Year", "Fiscal Period", "Publish Date"]

        # Merge on available keys
        fundamentals = income.merge(
            balance,
            on=[k for k in merge_keys if k in balance.columns],
            how="outer",
            suffixes=("", "_bal"),
        ).merge(
            cashflow,
            on=[k for k in merge_keys if k in cashflow.columns],
            how="outer",
            suffixes=("", "_cf"),
        )

        # Sort by fiscal period for TTM calculations
        if "Fiscal Year" in fundamentals.columns and "Fiscal Period" in fundamentals.columns:
            fundamentals = fundamentals.sort_values(
                ["Ticker", "Fiscal Year", "Fiscal Period"]
            )
        else:
            fundamentals = fundamentals.sort_values(["Ticker", "Publish Date"])

        logger.info(f"Merged EODHD fundamentals: {len(fundamentals)} records")
        return fundamentals

    def get_sector_mapping(self) -> pd.DataFrame:
        """
        Get ticker to sector mapping.

        First tries companies_detail.parquet (has sector/industry from fundamentals API),
        then falls back to companies.parquet (exchange symbol list - usually no sector).

        Returns:
            DataFrame with TICKER and sector columns
        """
        # First try the detailed company info from fundamentals API
        companies_detail_path = self.data_dir / "companies_detail.parquet"
        if companies_detail_path.exists():
            companies = pd.read_parquet(companies_detail_path)
            logger.info(f"Loading sector info from companies_detail.parquet ({len(companies)} companies)")

            # Find sector column
            sector_col = None
            for col in ["gics_sector", "sector", "Sector", "GICS Sector"]:
                if col in companies.columns and companies[col].notna().any():
                    sector_col = col
                    break

            if sector_col:
                ticker_col = "code" if "code" in companies.columns else "Code"
                mapping = companies[[ticker_col, sector_col]].copy()
                mapping = mapping.rename(columns={ticker_col: "TICKER", sector_col: "sector"})
                mapping = mapping.dropna(subset=["sector"])
                logger.info(f"Loaded sector mapping for {len(mapping)} tickers")
                return mapping

        # Fall back to basic companies file
        companies = self._companies_cache
        if companies is None:
            companies = self.load_companies()

        if companies.empty:
            return pd.DataFrame(columns=["TICKER", "sector"])

        # Find sector column - EODHD exchange list usually doesn't have it
        sector_col = None
        for col in ["GICS Sector", "Sector", "GicSector", "sector"]:
            if col in companies.columns:
                sector_col = col
                break

        if sector_col is None:
            logger.warning(
                "No sector information available. "
                "Re-download fundamentals to get sector/industry data from companies_detail.parquet"
            )
            return pd.DataFrame(columns=["TICKER", "sector"])

        ticker_col = "Ticker" if "Ticker" in companies.columns else "Code"
        mapping = companies[[ticker_col, sector_col]].copy()
        mapping = mapping.rename(columns={ticker_col: "TICKER", sector_col: "sector"})

        return mapping

    def get_financial_tickers(self) -> Set[str]:
        """
        Get set of financial company tickers to exclude.

        First tries companies_detail.parquet (has sector/industry from fundamentals API),
        then falls back to companies.parquet.

        Returns:
            Set of ticker symbols for financial companies
        """
        # First try the detailed company info
        companies_detail_path = self.data_dir / "companies_detail.parquet"
        if companies_detail_path.exists():
            companies = pd.read_parquet(companies_detail_path)
            ticker_col = "code" if "code" in companies.columns else "Code"

            financial_tickers = set()
            for _, row in companies.iterrows():
                sector = row.get("gics_sector") or row.get("sector")
                industry = row.get("industry")

                if is_financial_company(sector=sector, industry=industry):
                    financial_tickers.add(row[ticker_col])

            if financial_tickers:
                logger.info(f"Identified {len(financial_tickers)} financial tickers from companies_detail")
                return financial_tickers

        # Fall back to basic companies file
        companies = self._companies_cache
        if companies is None:
            companies = self.load_companies()

        if companies.empty:
            return set()

        ticker_col = "Ticker" if "Ticker" in companies.columns else "Code"

        # Check if we have sector/industry columns
        has_sector_info = any(
            col in companies.columns
            for col in ["Sector", "GICS Sector", "GicSector", "Industry", "GICS Industry"]
        )

        if not has_sector_info:
            logger.warning(
                "No sector information available to identify financial companies. "
                "Re-download fundamentals to get sector/industry data."
            )
            return set()

        financial_tickers = set()
        for _, row in companies.iterrows():
            sector = row.get("Sector") or row.get("GICS Sector") or row.get("GicSector")
            industry = row.get("Industry") or row.get("GICS Industry")

            if is_financial_company(sector=sector, industry=industry):
                financial_tickers.add(row[ticker_col])

        logger.info(f"Identified {len(financial_tickers)} financial tickers")
        return financial_tickers

    def get_source_name(self) -> str:
        """Return the data source name."""
        return "eodhd"

    def load_macro(self) -> pd.DataFrame:
        """
        Load macroeconomic data from EODHD cache.

        Returns:
            DataFrame with macro indicators indexed by date
        """
        filepath = self.data_dir / EODHD_FILES["macro"]
        if not filepath.exists():
            logger.warning(f"EODHD macro file not found: {filepath}")
            return pd.DataFrame()

        df = pd.read_parquet(filepath)

        # Pivot indicators to columns
        if "indicator" in df.columns and "Value" in df.columns:
            df["Date"] = pd.to_datetime(df.get("Date") or df.get("date"))
            df = df.pivot(index="Date", columns="indicator", values="Value")
            df = df.reset_index()
            df = df.rename(columns={"Date": "public_date"})

        logger.info(f"Loaded {len(df)} EODHD macro records")
        return df

    def get_shares_outstanding(self) -> pd.DataFrame:
        """
        Get shares outstanding data from fundamentals.

        EODHD provides shares outstanding in balance sheet data.

        Returns:
            DataFrame with Ticker, Date, Shares Outstanding
        """
        try:
            balance = self.load_balance()
        except FileNotFoundError:
            logger.warning("Balance sheet not found, cannot get shares outstanding")
            return pd.DataFrame(columns=["Ticker", "Date", "Shares Outstanding"])

        shares_col = None
        for col in ["commonStockSharesOutstanding", "sharesOutstanding", "Shares Outstanding"]:
            if col in balance.columns:
                shares_col = col
                break

        if shares_col is None:
            logger.warning(
                "Shares outstanding column not found in balance sheet. "
                "Re-download fundamentals with updated downloader to include this data."
            )
            return pd.DataFrame(columns=["Ticker", "Date", "Shares Outstanding"])

        result = balance[["Ticker", "Publish Date", shares_col]].copy()
        result = result.rename(columns={
            "Publish Date": "Date",
            shares_col: "Shares Outstanding"
        })

        # Ensure Date is datetime
        result["Date"] = pd.to_datetime(result["Date"], errors="coerce")

        # Drop rows with missing critical data
        result = result.dropna(subset=["Ticker", "Date", "Shares Outstanding"])

        logger.info(f"Loaded shares outstanding for {result['Ticker'].nunique()} tickers")
        return result
