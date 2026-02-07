"""
Benchmark data management.

Handles fetching and caching of S&P 500 benchmark data
to avoid repeated API calls during backtesting.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ..config import BENCHMARK_CACHE, CACHE_DIR

logger = logging.getLogger(__name__)


class BenchmarkManager:
    """
    Manages benchmark (S&P 500) data with caching.

    Fetches S&P 500 prices from yfinance and caches locally
    to avoid repeated downloads during backtest iterations.

    Example:
        bm = BenchmarkManager()
        returns = bm.get_returns(start_date, end_date, freq='Q')
    """

    def __init__(self, cache_path: Optional[Path] = None):
        """
        Initialize the benchmark manager.

        Args:
            cache_path: Path for cached benchmark data
        """
        self.cache_path = Path(cache_path) if cache_path else BENCHMARK_CACHE
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)

        self._data: Optional[pd.DataFrame] = None

    def fetch_data(
        self,
        start: datetime,
        end: Optional[datetime] = None,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """
        Fetch S&P 500 data with caching.

        Args:
            start: Start date
            end: End date (defaults to today)
            force_refresh: Force re-download even if cached

        Returns:
            DataFrame with Date index and Adj Close column
        """
        if end is None:
            end = datetime.now()

        # Check cache
        if not force_refresh and self._data is not None:
            return self._filter_dates(self._data, start, end)

        if not force_refresh and self.cache_path.exists():
            try:
                self._data = pd.read_parquet(self.cache_path)
                self._data.index = pd.to_datetime(self._data.index)

                # Check if cache covers requested range
                if self._data.index.min() <= start and self._data.index.max() >= end:
                    logger.info("Using cached S&P 500 data")
                    return self._filter_dates(self._data, start, end)
                else:
                    logger.info("Cache doesn't cover full date range, refreshing")
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")

        # Fetch from yfinance
        logger.info("Downloading S&P 500 data from yfinance...")
        try:
            import yfinance as yf

            # Fetch extra buffer for safety
            fetch_start = start - pd.DateOffset(months=6)
            # Use total return index (includes reinvested dividends)
            # for fair comparison against portfolio returns
            sp500 = yf.download(
                "^SP500TR", start=fetch_start, end=end, progress=False
            )

            if sp500.empty:
                logger.warning("Failed to download S&P 500 data")
                return pd.DataFrame()

            # Handle potential MultiIndex columns
            if isinstance(sp500.columns, pd.MultiIndex):
                sp500 = sp500.droplevel(1, axis=1)

            # Keep only Adj Close
            if "Adj Close" in sp500.columns:
                self._data = sp500[["Adj Close"]].copy()
            else:
                self._data = sp500[["Close"]].rename(columns={"Close": "Adj Close"})

            # Save to cache
            self._data.to_parquet(self.cache_path)
            logger.info(f"Cached S&P 500 data: {len(self._data)} days")

            return self._filter_dates(self._data, start, end)

        except ImportError:
            logger.error("yfinance not installed")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Failed to download benchmark data: {e}")
            return pd.DataFrame()

    def _filter_dates(
        self, data: pd.DataFrame, start: datetime, end: datetime
    ) -> pd.DataFrame:
        """Filter data to date range."""
        mask = (data.index >= start) & (data.index <= end)
        return data[mask].copy()

    def get_price_at_date(self, date: datetime) -> Optional[float]:
        """
        Get S&P 500 price at a specific date.

        Uses nearest available trading day.

        Args:
            date: Target date

        Returns:
            Price or None if not available
        """
        if self._data is None or self._data.empty:
            self.fetch_data(date - pd.DateOffset(years=1), date)

        if self._data is None or self._data.empty:
            return None

        try:
            idx = self._data.index.get_indexer([date], method="nearest")[0]
            return float(self._data.iloc[idx]["Adj Close"])
        except (IndexError, KeyError):
            return None

    def get_return(
        self, start_date: datetime, end_date: datetime
    ) -> Optional[float]:
        """
        Calculate return between two dates.

        Args:
            start_date: Period start
            end_date: Period end

        Returns:
            Return as decimal or None
        """
        start_price = self.get_price_at_date(start_date)
        end_price = self.get_price_at_date(end_date)

        if start_price is None or end_price is None or start_price <= 0:
            return None

        return (end_price - start_price) / start_price

    def get_period_returns(
        self,
        dates: list,
        holding_months: int = 3,
    ) -> pd.Series:
        """
        Get returns for multiple periods.

        Args:
            dates: List of period start dates
            holding_months: Holding period in months

        Returns:
            Series of returns indexed by start date
        """
        returns = {}

        for date in dates:
            end_date = date + pd.DateOffset(months=holding_months)
            ret = self.get_return(date, end_date)
            if ret is not None:
                returns[date] = ret

        return pd.Series(returns)

    def get_cumulative_returns(
        self, start: datetime, end: datetime
    ) -> pd.Series:
        """
        Get daily cumulative returns.

        Args:
            start: Start date
            end: End date

        Returns:
            Series of cumulative returns
        """
        data = self.fetch_data(start, end)
        if data.empty:
            return pd.Series()

        prices = data["Adj Close"]
        return (prices / prices.iloc[0]) - 1

    def get_monthly_returns(
        self, start: datetime, end: datetime
    ) -> pd.Series:
        """
        Get monthly returns.

        Args:
            start: Start date
            end: End date

        Returns:
            Series of monthly returns
        """
        data = self.fetch_data(start, end)
        if data.empty:
            return pd.Series()

        # Resample to month-end
        monthly = data["Adj Close"].resample("ME").last()
        return monthly.pct_change().dropna()

    def get_quarterly_returns(
        self, start: datetime, end: datetime
    ) -> pd.Series:
        """
        Get quarterly returns.

        Args:
            start: Start date
            end: End date

        Returns:
            Series of quarterly returns
        """
        data = self.fetch_data(start, end)
        if data.empty:
            return pd.Series()

        # Resample to quarter-end
        quarterly = data["Adj Close"].resample("QE").last()
        return quarterly.pct_change().dropna()

    def clear_cache(self) -> None:
        """Clear the cached benchmark data."""
        if self.cache_path.exists():
            self.cache_path.unlink()
        self._data = None
        logger.info("Cleared benchmark cache")
