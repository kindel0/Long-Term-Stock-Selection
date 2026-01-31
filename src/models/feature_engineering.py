"""
Feature engineering utilities for stock selection.

This module provides tools for preparing and transforming features
for the stock selection models.
"""

import logging
from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd

from ..config import FEATURE_CATEGORIES, EPS

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Feature engineering for stock selection.

    Provides methods for:
    - Computing financial ratios from raw data
    - TTM (trailing twelve months) calculations
    - Feature transformations and combinations
    - Data quality checks
    """

    def __init__(self, winsorize_mad: float = 5.0):
        """
        Initialize the feature engineer.

        Args:
            winsorize_mad: Number of MADs for winsorization (default 5)
        """
        self.winsorize_mad = winsorize_mad
        self._ratio_columns: List[str] = []

    def compute_ttm(
        self,
        df: pd.DataFrame,
        ticker_col: str = "TICKER",
        flow_columns: Optional[List[str]] = None,
        periods: int = 4,
    ) -> pd.DataFrame:
        """
        Compute trailing twelve month values for flow variables.

        Args:
            df: DataFrame with flow data (must be sorted by ticker and date)
            ticker_col: Column name for ticker identifier
            flow_columns: List of columns to compute TTM for
            periods: Number of periods to sum (default 4 for quarterly)

        Returns:
            DataFrame with added TTM columns
        """
        if flow_columns is None:
            # Default flow columns that typically need TTM treatment
            flow_columns = [
                "Revenue",
                "Cost of Revenue",
                "Gross Profit",
                "Operating Income (Loss)",
                "Net Income",
                "Net Cash from Operating Activities",
                "Depreciation & Amortization",
                "Interest Expense, Net",
            ]

        df = df.copy()

        for col in flow_columns:
            if col not in df.columns:
                continue

            ttm_col = f"{col}_ttm"
            df[ttm_col] = (
                df.groupby(ticker_col)[col]
                .rolling(periods, min_periods=periods)
                .sum()
                .reset_index(level=0, drop=True)
            )
            logger.debug(f"Computed TTM for {col}")

        return df

    def compute_lagged_averages(
        self,
        df: pd.DataFrame,
        ticker_col: str = "TICKER",
        columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Compute lagged averages for balance sheet items.

        Used for turnover ratios that require average denominators.

        Args:
            df: DataFrame with balance sheet data
            ticker_col: Column for grouping
            columns: Columns to compute averages for

        Returns:
            DataFrame with averaged columns
        """
        if columns is None:
            columns = [
                "Total Assets",
                "Total Equity",
                "Inventories",
                "Accounts & Notes Receivable",
                "Payables & Accruals",
            ]

        df = df.copy()

        for col in columns:
            if col not in df.columns:
                continue

            lag_col = f"{col}_lag1"
            avg_col = f"{col}_avg"

            df[lag_col] = df.groupby(ticker_col)[col].shift(1)
            df[avg_col] = (df[col] + df[lag_col]) / 2

        return df

    def compute_financial_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all financial ratios from raw data.

        This replicates the ratio calculations from improved_simfin_panel.py.

        Args:
            df: DataFrame with required fundamental columns

        Returns:
            DataFrame with computed ratio columns
        """
        df = df.copy()
        eps = EPS

        # Helper for safe division
        def safe_div(a, b):
            result = np.where(np.abs(b) < eps, np.nan, a / b)
            return pd.Series(result, index=a.index if hasattr(a, "index") else None)

        # ----- Valuation Ratios -----
        if "Total Equity" in df.columns and "MthCap" in df.columns:
            df["bm"] = df["Total Equity"] / (df["MthCap"] + eps)
            df["ptb"] = (df["MthCap"] + eps) / (df["Total Equity"] + eps)

        if "Revenue_ttm" in df.columns and "MthCap" in df.columns:
            df["ps"] = (df["MthCap"] + eps) / (df["Revenue_ttm"] + eps)

        if "Net Cash from Operating Activities_ttm" in df.columns:
            df["pcf"] = (df["MthCap"] + eps) / (
                df["Net Cash from Operating Activities_ttm"] + eps
            )

        # ----- Profitability Ratios -----
        if "Net Income_ttm" in df.columns:
            if "Revenue_ttm" in df.columns:
                df["npm"] = df["Net Income_ttm"] / (df["Revenue_ttm"] + eps)

            if "Total Assets" in df.columns:
                df["roa"] = df["Net Income_ttm"] / (df["Total Assets"] + eps)

            if "Total Equity" in df.columns:
                df["roe"] = df["Net Income_ttm"] / (df["Total Equity"] + eps)

        if "Gross Profit_ttm" in df.columns and "Revenue_ttm" in df.columns:
            df["gpm"] = df["Gross Profit_ttm"] / (df["Revenue_ttm"] + eps)

        # ----- Solvency Ratios -----
        if "total_debt" not in df.columns:
            if "Short Term Debt" in df.columns and "Long Term Debt" in df.columns:
                df["total_debt"] = (
                    df["Short Term Debt"].fillna(0) + df["Long Term Debt"].fillna(0)
                )

        if "total_debt" in df.columns:
            if "Total Equity" in df.columns:
                df["de_ratio"] = df["total_debt"] / (df["Total Equity"] + eps)

            if "Total Assets" in df.columns:
                df["debt_at"] = df["total_debt"] / (df["Total Assets"] + eps)
                df["debt_assets"] = df["debt_at"]

        # ----- Liquidity Ratios -----
        if (
            "Total Current Assets" in df.columns
            and "Total Current Liabilities" in df.columns
        ):
            df["curr_ratio"] = df["Total Current Assets"] / (
                df["Total Current Liabilities"] + eps
            )

            if "Inventories" in df.columns:
                df["quick_ratio"] = (
                    df["Total Current Assets"] - df["Inventories"]
                ) / (df["Total Current Liabilities"] + eps)

        if "Cash, Cash Equivalents & Short Term Investments" in df.columns:
            if "Total Current Liabilities" in df.columns:
                df["cash_ratio"] = df[
                    "Cash, Cash Equivalents & Short Term Investments"
                ] / (df["Total Current Liabilities"] + eps)

        # ----- Efficiency Ratios -----
        if "Revenue_ttm" in df.columns and "Total Assets_avg" in df.columns:
            df["at_turn"] = df["Revenue_ttm"] / (df["Total Assets_avg"] + eps)

        if "Cost of Revenue_ttm" in df.columns:
            if "Inventories_avg" in df.columns:
                df["inv_turn"] = df["Cost of Revenue_ttm"] / (
                    df["Inventories_avg"] + eps
                )

        if "Revenue_ttm" in df.columns:
            if "Accounts & Notes Receivable_avg" in df.columns:
                df["rect_turn"] = df["Revenue_ttm"] / (
                    df["Accounts & Notes Receivable_avg"] + eps
                )

        # Store ratio columns for later reference
        self._ratio_columns = [
            col
            for col in df.columns
            if any(
                keyword in col.lower()
                for keyword in [
                    "ratio",
                    "turn",
                    "margin",
                    "return",
                    "_at",
                    "_eq",
                    "bm",
                    "ptb",
                    "ps",
                    "pcf",
                    "pe_",
                    "npm",
                    "gpm",
                    "roa",
                    "roe",
                ]
            )
        ]

        return df

    def winsorize_ratios(
        self, df: pd.DataFrame, columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Winsorize ratio columns at median +/- n*MAD.

        Args:
            df: DataFrame with ratio columns
            columns: Columns to winsorize (uses auto-detected if None)

        Returns:
            DataFrame with winsorized ratios
        """
        if columns is None:
            columns = self._ratio_columns or self._detect_ratio_columns(df)

        df = df.copy()

        for col in columns:
            if col not in df.columns:
                continue

            series = df[col]
            median = series.median()
            mad = (series - median).abs().median()

            if pd.isna(mad) or mad == 0:
                continue

            lower = median - self.winsorize_mad * mad
            upper = median + self.winsorize_mad * mad

            df[col] = series.clip(lower=lower, upper=upper)

        return df

    def _detect_ratio_columns(self, df: pd.DataFrame) -> List[str]:
        """Detect columns that are likely financial ratios."""
        ratio_keywords = [
            "ratio",
            "turn",
            "ptb",
            "ps",
            "pcf",
            "pe_",
            "bm",
            "npm",
            "gpm",
            "roa",
            "roe",
            "evm",
            "dpr",
            "opm",
            "ptpm",
            "cfm",
            "roce",
            "efftax",
            "debt_",
            "int_",
            "profit_",
            "ocf_",
            "cash_",
            "fcf_",
            "accrual",
            "sale_",
            "rd_sale",
        ]

        return [
            col
            for col in df.columns
            if any(keyword in col.lower() for keyword in ratio_keywords)
        ]

    def calculate_forward_returns(
        self,
        df: pd.DataFrame,
        price_col: str = "MthPrc",
        ticker_col: str = "TICKER",
        date_col: str = "public_date",
        periods: int = 3,
        validate_gap: bool = True,
    ) -> pd.DataFrame:
        """
        Calculate forward returns with date validation.

        Args:
            df: DataFrame sorted by ticker and date
            price_col: Column with prices
            ticker_col: Column for ticker grouping
            date_col: Column with dates
            periods: Number of periods forward
            validate_gap: Whether to validate time gap

        Returns:
            DataFrame with forward return column
        """
        df = df.copy()
        df = df.sort_values([ticker_col, date_col])

        # Calculate future price and date
        df["_future_price"] = df.groupby(ticker_col)[price_col].shift(-periods)
        df["_future_date"] = df.groupby(ticker_col)[date_col].shift(-periods)

        # Calculate return
        return_col = f"{periods}mo_return"
        df[return_col] = (df["_future_price"] - df[price_col]) / (df[price_col] + EPS)

        # Validate time gap if requested
        if validate_gap:
            days_gap = (df["_future_date"] - df[date_col]).dt.days
            expected_days = periods * 30  # Approximate
            tolerance = expected_days * 0.25  # 25% tolerance

            invalid_gap = (days_gap < expected_days - tolerance) | (
                days_gap > expected_days + tolerance
            )
            df.loc[invalid_gap, return_col] = np.nan

        # Handle invalid prices
        df.loc[df[price_col] <= 0, return_col] = np.nan

        # Clean up
        df = df.drop(columns=["_future_price", "_future_date"])

        return df

    def add_outcome_date(
        self,
        df: pd.DataFrame,
        date_col: str = "public_date",
        months_forward: int = 3,
        reporting_lag_days: int = 3,
    ) -> pd.DataFrame:
        """
        Add outcome date for PIT compliance.

        The outcome date is when we would have known the forward return,
        accounting for reporting lag.

        Args:
            df: DataFrame with date column
            date_col: Column with base dates
            months_forward: How many months forward the return is
            reporting_lag_days: Buffer for reporting lag

        Returns:
            DataFrame with 'outcome_date' column
        """
        df = df.copy()
        df["outcome_date"] = (
            pd.to_datetime(df[date_col])
            + pd.DateOffset(months=months_forward)
            + pd.DateOffset(days=reporting_lag_days)
        )
        return df

    def get_available_features(
        self, df: pd.DataFrame, categories: Optional[List[str]] = None
    ) -> Dict[str, List[str]]:
        """
        Get available features by category.

        Args:
            df: DataFrame to check
            categories: Categories to check (None for all)

        Returns:
            Dict mapping category names to available feature lists
        """
        if categories is None:
            categories = list(FEATURE_CATEGORIES.keys())

        available = {}
        for cat in categories:
            if cat in FEATURE_CATEGORIES:
                available[cat] = [
                    f for f in FEATURE_CATEGORIES[cat] if f in df.columns
                ]

        return available
