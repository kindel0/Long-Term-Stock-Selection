"""
Abstract base class for data loaders.

Defines the interface that all data source loaders must implement,
enabling easy switching between SimFin, EODHD, or other providers.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Set

import pandas as pd


class DataLoader(ABC):
    """
    Abstract base class for loading financial data.

    All data loaders (SimFin, EODHD, etc.) must implement this interface
    to be compatible with the PanelBuilder.

    The returned DataFrames should use standardized column names:
    - Prices: TICKER, Date, Adj. Close, Dividend, Shares Outstanding
    - Fundamentals: TICKER, Publish Date, plus standard financial fields
    """

    def __init__(self, data_dir: Optional[Path] = None):
        """
        Initialize the loader.

        Args:
            data_dir: Directory containing data files
        """
        self.data_dir = Path(data_dir) if data_dir else None

    @abstractmethod
    def load_prices(self) -> pd.DataFrame:
        """
        Load share price data.

        Returns:
            DataFrame with columns:
            - Ticker: Stock symbol
            - SimFinId/gvkey: Unique identifier (optional)
            - Date: Trading date
            - Adj. Close: Adjusted closing price
            - Dividend: Dividend amount (0 if none)
            - Shares Outstanding: Number of shares
        """
        pass

    @abstractmethod
    def load_income(self) -> pd.DataFrame:
        """
        Load income statement data.

        Returns:
            DataFrame with income statement line items and:
            - Ticker: Stock symbol
            - Fiscal Year: Fiscal year
            - Fiscal Period: Q1, Q2, Q3, Q4
            - Publish Date: When data was publicly available (for PIT)
        """
        pass

    @abstractmethod
    def load_balance(self) -> pd.DataFrame:
        """
        Load balance sheet data.

        Returns:
            DataFrame with balance sheet line items and:
            - Ticker: Stock symbol
            - Fiscal Year: Fiscal year
            - Fiscal Period: Q1, Q2, Q3, Q4
            - Publish Date: When data was publicly available (for PIT)
        """
        pass

    @abstractmethod
    def load_cashflow(self) -> pd.DataFrame:
        """
        Load cash flow statement data.

        Returns:
            DataFrame with cash flow line items and:
            - Ticker: Stock symbol
            - Fiscal Year: Fiscal year
            - Fiscal Period: Q1, Q2, Q3, Q4
            - Publish Date: When data was publicly available (for PIT)
        """
        pass

    @abstractmethod
    def load_fundamentals(self) -> pd.DataFrame:
        """
        Load and merge all fundamental data (income, balance, cashflow).

        Returns:
            Merged DataFrame with all fundamental data
        """
        pass

    @abstractmethod
    def load_companies(self) -> pd.DataFrame:
        """
        Load company metadata.

        Returns:
            DataFrame with company info including:
            - Ticker: Stock symbol
            - Company Name: Full company name
            - IndustryId or Industry: Industry classification
        """
        pass

    @abstractmethod
    def get_sector_mapping(self) -> pd.DataFrame:
        """
        Get ticker to sector mapping.

        Returns:
            DataFrame with columns:
            - TICKER: Stock symbol
            - sector: Sector name
        """
        pass

    @abstractmethod
    def get_financial_tickers(self) -> Set[str]:
        """
        Get set of financial company tickers to exclude.

        Returns:
            Set of ticker symbols for financial companies
        """
        pass

    def get_source_name(self) -> str:
        """
        Get the name of this data source.

        Returns:
            Name string (e.g., 'simfin', 'eodhd')
        """
        return self.__class__.__name__.lower().replace("loader", "")

    def get_shares_outstanding(self) -> pd.DataFrame:
        """
        Get shares outstanding data, if not already in price data.

        This is optional - only needed if load_prices() doesn't include
        Shares Outstanding (e.g., EODHD). SimFin includes it in prices.

        Returns:
            DataFrame with columns: Ticker, Date, Shares Outstanding
            Or empty DataFrame if shares are in price data
        """
        # Default: return empty - assume shares are in price data
        return pd.DataFrame(columns=["Ticker", "Date", "Shares Outstanding"])
