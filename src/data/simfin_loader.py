"""
SimFin data loading utilities.

Handles loading and parsing of SimFin CSV files with proper
column selection and type conversion.
"""

import logging
from pathlib import Path
from typing import List, Optional, Set

import pandas as pd

from ..config import (
    SIMFIN_DIR,
    SIMFIN_PRICE_COLS,
    SIMFIN_INCOME_COLS,
    SIMFIN_BALANCE_COLS,
    SIMFIN_CASHFLOW_COLS,
)

logger = logging.getLogger(__name__)


class SimFinLoader:
    """
    Loader for SimFin data files.

    Handles the semicolon-delimited CSV format used by SimFin
    and provides methods for loading each data type.
    """

    def __init__(self, data_dir: Optional[Path] = None):
        """
        Initialize the loader.

        Args:
            data_dir: Directory containing SimFin CSV files (default from config)
        """
        self.data_dir = Path(data_dir) if data_dir else SIMFIN_DIR

    def _read_simfin(
        self, filename: str, usecols: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Read a SimFin CSV file.

        Args:
            filename: Name of the file (relative to data_dir)
            usecols: Columns to load (loads all if None)

        Returns:
            DataFrame with loaded data
        """
        filepath = self.data_dir / filename

        if not filepath.exists():
            raise FileNotFoundError(f"SimFin file not found: {filepath}")

        # Get available columns
        available = self._get_available_columns(filepath)

        # Filter requested columns to available ones
        if usecols:
            usecols = [c for c in usecols if c in available]

        df = pd.read_csv(filepath, sep=";", usecols=usecols, low_memory=False)

        logger.debug(f"Loaded {len(df)} rows from {filename}")
        return df

    def _get_available_columns(self, filepath: Path) -> Set[str]:
        """Get the set of available columns in a file."""
        return set(pd.read_csv(filepath, sep=";", nrows=0).columns)

    def _to_numeric(self, series: pd.Series) -> pd.Series:
        """Convert a series to numeric, coercing errors to NaN."""
        return pd.to_numeric(series, errors="coerce")

    def load_prices(self) -> pd.DataFrame:
        """
        Load share price data.

        Returns:
            DataFrame with columns: Ticker, SimFinId, Date, Adj. Close,
            Dividend, Shares Outstanding
        """
        df = self._read_simfin("us-shareprices-daily.csv", usecols=SIMFIN_PRICE_COLS)

        # Parse dates
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Ticker", "SimFinId", "Date"])

        # Convert numerics
        df["Adj. Close"] = self._to_numeric(df["Adj. Close"])
        df["Dividend"] = self._to_numeric(df["Dividend"]).fillna(0.0)
        df["Shares Outstanding"] = self._to_numeric(df["Shares Outstanding"])

        df = df.sort_values(["Ticker", "Date"])

        logger.info(f"Loaded {len(df)} price records for {df['Ticker'].nunique()} tickers")
        return df

    def load_income(self) -> pd.DataFrame:
        """
        Load income statement data.

        Returns:
            DataFrame with income statement line items
        """
        df = self._read_simfin("us-income-quarterly.csv", usecols=SIMFIN_INCOME_COLS)

        df["Publish Date"] = pd.to_datetime(df["Publish Date"], errors="coerce")
        df = df.dropna(subset=["Ticker", "Publish Date"])

        # Convert numeric columns
        for col in df.columns:
            if col not in ["Ticker", "Fiscal Year", "Fiscal Period", "Publish Date"]:
                df[col] = self._to_numeric(df[col])

        logger.info(f"Loaded {len(df)} income records")
        return df

    def load_balance(self) -> pd.DataFrame:
        """
        Load balance sheet data.

        Returns:
            DataFrame with balance sheet line items
        """
        df = self._read_simfin("us-balance-quarterly.csv", usecols=SIMFIN_BALANCE_COLS)

        df["Publish Date"] = pd.to_datetime(df["Publish Date"], errors="coerce")
        df = df.dropna(subset=["Ticker", "Publish Date"])

        # Convert numeric columns
        for col in df.columns:
            if col not in ["Ticker", "Fiscal Year", "Fiscal Period", "Publish Date"]:
                df[col] = self._to_numeric(df[col])

        logger.info(f"Loaded {len(df)} balance sheet records")
        return df

    def load_cashflow(self) -> pd.DataFrame:
        """
        Load cash flow statement data.

        Returns:
            DataFrame with cash flow line items
        """
        df = self._read_simfin("us-cashflow-quarterly.csv", usecols=SIMFIN_CASHFLOW_COLS)

        df["Publish Date"] = pd.to_datetime(df["Publish Date"], errors="coerce")
        df = df.dropna(subset=["Ticker", "Publish Date"])

        # Convert numeric columns
        for col in df.columns:
            if col not in ["Ticker", "Fiscal Year", "Fiscal Period", "Publish Date"]:
                df[col] = self._to_numeric(df[col])

        logger.info(f"Loaded {len(df)} cash flow records")
        return df

    def load_companies(self) -> pd.DataFrame:
        """
        Load company metadata.

        Returns:
            DataFrame with company info including IndustryId
        """
        filepath = self.data_dir / "us-companies.csv"
        if not filepath.exists():
            logger.warning("us-companies.csv not found")
            return pd.DataFrame()

        df = pd.read_csv(filepath, sep=";")
        logger.info(f"Loaded {len(df)} company records")
        return df

    def load_industries(self) -> pd.DataFrame:
        """
        Load industry classification data.

        Returns:
            DataFrame with industry and sector mappings
        """
        filepath = self.data_dir / "industries.csv"
        if not filepath.exists():
            logger.warning("industries.csv not found")
            return pd.DataFrame()

        df = pd.read_csv(filepath, sep=";")
        logger.info(f"Loaded {len(df)} industry records")
        return df

    def load_fundamentals(self) -> pd.DataFrame:
        """
        Load and merge all fundamental data (income, balance, cashflow).

        Returns:
            Merged DataFrame with all fundamental data
        """
        income = self.load_income()
        balance = self.load_balance()
        cashflow = self.load_cashflow()

        # Merge on fiscal keys
        fundamentals = income.merge(
            balance,
            on=["Ticker", "Fiscal Year", "Fiscal Period", "Publish Date"],
            how="outer",
        ).merge(
            cashflow,
            on=["Ticker", "Fiscal Year", "Fiscal Period", "Publish Date"],
            how="outer",
        )

        # Sort by fiscal period for TTM calculations
        fundamentals = fundamentals.sort_values(
            ["Ticker", "Fiscal Year", "Fiscal Period"]
        )

        logger.info(f"Merged fundamentals: {len(fundamentals)} records")
        return fundamentals

    def get_sector_mapping(self) -> pd.DataFrame:
        """
        Get ticker to sector mapping.

        Returns:
            DataFrame with TICKER and sector columns
        """
        companies = self.load_companies()
        industries = self.load_industries()

        if companies.empty or industries.empty:
            return pd.DataFrame(columns=["TICKER", "sector"])

        merged = companies.merge(industries, on="IndustryId", how="left")
        mapping = merged[["Ticker", "Sector"]].rename(
            columns={"Ticker": "TICKER", "Sector": "sector"}
        )

        return mapping

    def get_financial_tickers(self) -> Set[str]:
        """
        Get set of financial company tickers to exclude.

        Returns:
            Set of ticker symbols for financial companies
        """
        companies = self.load_companies()
        if companies.empty:
            return set()

        # IndustryId starting with 103 = Financial Services
        financial = companies[
            companies["IndustryId"].astype(str).str.startswith("103")
        ]

        return set(financial["Ticker"])
