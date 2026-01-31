"""
Data validation utilities for stock selection.

Provides validation functions to ensure data quality and integrity.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..config import FEATURE_CATEGORIES, QUALITY_FILTERS

logger = logging.getLogger(__name__)


class DataValidator:
    """
    Validates stock selection data for quality and completeness.

    Provides checks for:
    - Required columns
    - Data types
    - Value ranges
    - Missing data patterns
    - Lookahead bias
    """

    def __init__(self):
        """Initialize the validator."""
        self.issues: List[str] = []
        self.warnings: List[str] = []

    def validate_panel(
        self, df: pd.DataFrame, strict: bool = False
    ) -> Tuple[bool, Dict]:
        """
        Validate a panel dataset.

        Args:
            df: Panel DataFrame to validate
            strict: If True, warnings are treated as errors

        Returns:
            Tuple of (is_valid, report_dict)
        """
        self.issues = []
        self.warnings = []

        # Required columns
        self._check_required_columns(df)

        # Data types
        self._check_data_types(df)

        # Value ranges
        self._check_value_ranges(df)

        # Missing data
        self._check_missing_patterns(df)

        # Date ordering
        self._check_date_ordering(df)

        # Feature availability
        self._check_feature_availability(df)

        is_valid = len(self.issues) == 0
        if strict:
            is_valid = is_valid and len(self.warnings) == 0

        report = {
            "is_valid": is_valid,
            "issues": self.issues,
            "warnings": self.warnings,
            "shape": df.shape,
            "date_range": (
                str(df["public_date"].min()) if "public_date" in df.columns else None,
                str(df["public_date"].max()) if "public_date" in df.columns else None,
            ),
            "n_tickers": df["TICKER"].nunique() if "TICKER" in df.columns else 0,
        }

        return is_valid, report

    def _check_required_columns(self, df: pd.DataFrame) -> None:
        """Check for required columns."""
        required = ["TICKER", "public_date", "MthPrc", "MthCap"]

        missing = [col for col in required if col not in df.columns]
        if missing:
            self.issues.append(f"Missing required columns: {missing}")

    def _check_data_types(self, df: pd.DataFrame) -> None:
        """Check data types of key columns."""
        if "public_date" in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df["public_date"]):
                self.issues.append("public_date is not datetime type")

        numeric_cols = ["MthPrc", "MthCap", "1yr_return"]
        for col in numeric_cols:
            if col in df.columns:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    self.issues.append(f"{col} is not numeric type")

    def _check_value_ranges(self, df: pd.DataFrame) -> None:
        """Check value ranges for key columns."""
        if "MthPrc" in df.columns:
            neg_prices = (df["MthPrc"] < 0).sum()
            if neg_prices > 0:
                self.issues.append(f"{neg_prices} rows have negative prices")

            penny_stocks = (df["MthPrc"] < QUALITY_FILTERS["min_price"]).sum()
            if penny_stocks > 0:
                self.warnings.append(f"{penny_stocks} penny stock observations")

        if "MthCap" in df.columns:
            neg_cap = (df["MthCap"] <= 0).sum()
            if neg_cap > 0:
                self.issues.append(f"{neg_cap} rows have non-positive market cap")

        if "1yr_return" in df.columns:
            low, high = QUALITY_FILTERS["return_bounds"]
            extreme = (
                (df["1yr_return"] < low) | (df["1yr_return"] > high)
            ).sum()
            if extreme > 0:
                self.warnings.append(
                    f"{extreme} observations have extreme returns "
                    f"(outside [{low}, {high}])"
                )

    def _check_missing_patterns(self, df: pd.DataFrame) -> None:
        """Check missing data patterns."""
        # Overall missing rate
        missing_rate = df.isnull().sum().sum() / (len(df) * len(df.columns))
        if missing_rate > 0.3:
            self.warnings.append(f"High overall missing rate: {missing_rate:.1%}")

        # Target missing rate
        if "1yr_return" in df.columns:
            target_missing = df["1yr_return"].isnull().mean()
            if target_missing > 0.5:
                self.warnings.append(
                    f"High target missing rate: {target_missing:.1%}"
                )

        # Critical feature missing
        critical = ["roe", "roa", "ptb", "de_ratio"]
        for col in critical:
            if col in df.columns:
                missing = df[col].isnull().mean()
                if missing > 0.7:
                    self.warnings.append(
                        f"Critical feature {col} has {missing:.1%} missing"
                    )

    def _check_date_ordering(self, df: pd.DataFrame) -> None:
        """Check that dates are properly ordered."""
        if "TICKER" not in df.columns or "public_date" not in df.columns:
            return

        # Check for duplicate ticker-dates
        dups = df.duplicated(subset=["TICKER", "public_date"]).sum()
        if dups > 0:
            self.warnings.append(f"{dups} duplicate ticker-date combinations")

        # Check ordering within tickers
        sample_tickers = df["TICKER"].unique()[:10]
        for ticker in sample_tickers:
            ticker_df = df[df["TICKER"] == ticker]
            if not ticker_df["public_date"].is_monotonic_increasing:
                self.warnings.append(
                    f"Ticker {ticker} has non-monotonic dates"
                )
                break

    def _check_feature_availability(self, df: pd.DataFrame) -> None:
        """Check availability of feature categories."""
        for category, features in FEATURE_CATEGORIES.items():
            available = [f for f in features if f in df.columns]
            if len(available) == 0:
                self.warnings.append(f"No features available for category: {category}")
            elif len(available) < len(features) / 2:
                self.warnings.append(
                    f"Category {category}: only {len(available)}/{len(features)} features available"
                )

    def check_pit_compliance(
        self,
        df: pd.DataFrame,
        date_col: str = "public_date",
        outcome_col: str = "outcome_date",
        return_col: str = "1yr_return",
    ) -> Tuple[bool, Dict]:
        """
        Check for potential point-in-time violations.

        Args:
            df: DataFrame to check
            date_col: Column with observation dates
            outcome_col: Column with outcome dates
            return_col: Column with forward returns

        Returns:
            Tuple of (is_compliant, report_dict)
        """
        issues = []

        if outcome_col not in df.columns:
            # Can't check PIT without outcome date
            return True, {"note": "No outcome_date column to check"}

        # Check that outcome_date > public_date
        if date_col in df.columns and outcome_col in df.columns:
            violations = (df[outcome_col] <= df[date_col]).sum()
            if violations > 0:
                issues.append(
                    f"{violations} rows have outcome_date <= public_date"
                )

        # Check that returns align with expected holding period
        if return_col in df.columns and outcome_col in df.columns:
            holding_days = (df[outcome_col] - df[date_col]).dt.days
            expected = 365  # 1 year for 1yr_return
            tolerance = 30

            outliers = (
                (holding_days < expected - tolerance)
                | (holding_days > expected + tolerance)
            ).sum()
            if outliers > len(df) * 0.1:
                issues.append(
                    f"{outliers} rows have unexpected holding period "
                    f"(not near {expected} days)"
                )

        is_compliant = len(issues) == 0
        return is_compliant, {"is_compliant": is_compliant, "issues": issues}

    def get_data_summary(self, df: pd.DataFrame) -> Dict:
        """
        Get a summary of the dataset.

        Args:
            df: DataFrame to summarize

        Returns:
            Dictionary with summary statistics
        """
        summary = {
            "shape": df.shape,
            "memory_mb": df.memory_usage(deep=True).sum() / 1e6,
        }

        if "TICKER" in df.columns:
            summary["n_tickers"] = df["TICKER"].nunique()

        if "public_date" in df.columns:
            summary["date_range"] = (
                str(df["public_date"].min()),
                str(df["public_date"].max()),
            )
            summary["n_months"] = df["public_date"].nunique()

        if "sector" in df.columns:
            summary["sectors"] = df["sector"].value_counts().to_dict()

        if "cap" in df.columns:
            summary["cap_distribution"] = df["cap"].value_counts().to_dict()

        if "1yr_return" in df.columns:
            valid_returns = df["1yr_return"].dropna()
            summary["return_stats"] = {
                "mean": valid_returns.mean(),
                "std": valid_returns.std(),
                "min": valid_returns.min(),
                "max": valid_returns.max(),
                "valid_pct": len(valid_returns) / len(df),
            }

        # Feature availability
        all_features = [f for cat in FEATURE_CATEGORIES.values() for f in cat]
        available = [f for f in all_features if f in df.columns]
        summary["features"] = {
            "total_defined": len(all_features),
            "available": len(available),
            "missing": len(all_features) - len(available),
        }

        # Missing data summary
        missing_pct = df.isnull().mean()
        summary["missing"] = {
            "mean_missing_rate": missing_pct.mean(),
            "cols_over_50pct_missing": (missing_pct > 0.5).sum(),
            "cols_complete": (missing_pct == 0).sum(),
        }

        return summary

    def print_report(self, report: Dict) -> None:
        """Print a validation report to the console."""
        print("\n" + "=" * 60)
        print("DATA VALIDATION REPORT")
        print("=" * 60)

        print(f"\nValid: {report.get('is_valid', 'N/A')}")
        print(f"Shape: {report.get('shape', 'N/A')}")
        print(f"Date Range: {report.get('date_range', 'N/A')}")
        print(f"Tickers: {report.get('n_tickers', 'N/A')}")

        if report.get("issues"):
            print("\nISSUES:")
            for issue in report["issues"]:
                print(f"  - {issue}")

        if report.get("warnings"):
            print("\nWARNINGS:")
            for warning in report["warnings"]:
                print(f"  - {warning}")

        print("=" * 60)
