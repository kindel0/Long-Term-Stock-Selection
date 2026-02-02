"""
Data validation utilities for stock selection.

Provides validation functions to ensure data quality and integrity,
including comparison between different data sources (SimFin, EODHD).
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

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

            # Map return column to expected days
            expected_days = {
                "1mo_return": 30,
                "3mo_return": 91,
                "6mo_return": 182,
                "1yr_return": 365,
            }
            expected = expected_days.get(return_col, 365)
            tolerance = 15 if expected <= 91 else 30

            outliers = (
                (holding_days < expected - tolerance)
                | (holding_days > expected + tolerance)
            ).sum()
            if outliers > len(df) * 0.1:
                issues.append(
                    f"{outliers} rows have unexpected holding period "
                    f"(not near {expected} days for {return_col})"
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


class DataSourceComparator:
    """
    Compare data quality between different data sources (SimFin vs EODHD).

    Provides methods to compare:
    - Ticker overlap
    - Fundamental data correlation
    - Price data correlation
    - Missing data patterns
    """

    def __init__(self):
        """Initialize the comparator."""
        self.comparison_results = {}

    def compare_fundamentals(
        self,
        simfin_df: pd.DataFrame,
        eodhd_df: pd.DataFrame,
        key_metrics: Optional[List[str]] = None,
        sample_tickers: int = 100,
    ) -> Dict:
        """
        Compare fundamental data between two sources.

        Args:
            simfin_df: SimFin fundamentals DataFrame
            eodhd_df: EODHD fundamentals DataFrame
            key_metrics: Metrics to compare (default: common fundamentals)
            sample_tickers: Number of tickers to sample for comparison

        Returns:
            Comparison report dictionary
        """
        key_metrics = key_metrics or [
            "Revenue",
            "Net Income",
            "Total Assets",
            "Total Equity",
            "Net Cash from Operating Activities",
        ]

        # Find overlapping tickers
        simfin_tickers = set(simfin_df["Ticker"].unique())
        eodhd_tickers = set(eodhd_df["Ticker"].unique())
        overlap = simfin_tickers & eodhd_tickers

        report = {
            "simfin_tickers": len(simfin_tickers),
            "eodhd_tickers": len(eodhd_tickers),
            "overlapping_tickers": len(overlap),
            "overlap_pct": len(overlap) / max(len(simfin_tickers), 1) * 100,
            "metric_comparisons": {},
        }

        if len(overlap) == 0:
            report["error"] = "No overlapping tickers found"
            return report

        # Sample tickers for detailed comparison
        sample = list(overlap)[:sample_tickers]

        # Compare each metric
        for metric in key_metrics:
            if metric not in simfin_df.columns or metric not in eodhd_df.columns:
                report["metric_comparisons"][metric] = {"status": "missing_column"}
                continue

            metric_report = self._compare_metric(
                simfin_df, eodhd_df, metric, sample
            )
            report["metric_comparisons"][metric] = metric_report

        return report

    def _compare_metric(
        self,
        df1: pd.DataFrame,
        df2: pd.DataFrame,
        metric: str,
        tickers: List[str],
    ) -> Dict:
        """
        Compare a single metric between two data sources.

        Args:
            df1: First DataFrame (SimFin)
            df2: Second DataFrame (EODHD)
            metric: Column name to compare
            tickers: List of tickers to include

        Returns:
            Metric comparison dictionary
        """
        # Filter to sample tickers and get most recent value per ticker
        df1_filtered = df1[df1["Ticker"].isin(tickers)].copy()
        df2_filtered = df2[df2["Ticker"].isin(tickers)].copy()

        # Get most recent value per ticker
        if "Publish Date" in df1_filtered.columns:
            df1_latest = df1_filtered.sort_values("Publish Date").groupby("Ticker").last()
        else:
            df1_latest = df1_filtered.groupby("Ticker").last()

        if "Publish Date" in df2_filtered.columns:
            df2_latest = df2_filtered.sort_values("Publish Date").groupby("Ticker").last()
        else:
            df2_latest = df2_filtered.groupby("Ticker").last()

        # Merge on ticker index
        merged = df1_latest[[metric]].join(
            df2_latest[[metric]],
            lsuffix="_simfin",
            rsuffix="_eodhd",
            how="inner",
        )

        if len(merged) == 0:
            return {"status": "no_overlap", "n_compared": 0}

        col1 = f"{metric}_simfin"
        col2 = f"{metric}_eodhd"

        # Drop rows where either is NaN
        valid = merged.dropna(subset=[col1, col2])

        if len(valid) < 2:
            return {"status": "insufficient_data", "n_compared": len(valid)}

        # Calculate correlation
        correlation = valid[col1].corr(valid[col2])

        # Calculate mean absolute percentage difference
        diff = (valid[col1] - valid[col2]).abs()
        denom = (valid[col1].abs() + valid[col2].abs()) / 2
        mape = (diff / denom.replace(0, np.nan)).mean() * 100

        # Calculate match rate (within 10%)
        pct_diff = diff / denom.replace(0, np.nan)
        match_rate = (pct_diff < 0.10).mean() * 100

        return {
            "status": "compared",
            "n_compared": len(valid),
            "correlation": round(correlation, 4),
            "mape_pct": round(mape, 2),
            "match_rate_10pct": round(match_rate, 2),
        }

    def compare_prices(
        self,
        simfin_df: pd.DataFrame,
        eodhd_df: pd.DataFrame,
        sample_tickers: int = 50,
    ) -> Dict:
        """
        Compare price data between two sources.

        Args:
            simfin_df: SimFin price DataFrame
            eodhd_df: EODHD price DataFrame
            sample_tickers: Number of tickers to sample

        Returns:
            Price comparison report
        """
        # Find overlapping tickers
        simfin_tickers = set(simfin_df["Ticker"].unique())
        eodhd_tickers = set(eodhd_df["Ticker"].unique())
        overlap = list(simfin_tickers & eodhd_tickers)

        report = {
            "simfin_tickers": len(simfin_tickers),
            "eodhd_tickers": len(eodhd_tickers),
            "overlapping_tickers": len(overlap),
        }

        if len(overlap) == 0:
            report["error"] = "No overlapping tickers found"
            return report

        # Sample for comparison
        sample = overlap[:sample_tickers]

        correlations = []
        for ticker in sample:
            sf = simfin_df[simfin_df["Ticker"] == ticker].copy()
            eo = eodhd_df[eodhd_df["Ticker"] == ticker].copy()

            if len(sf) < 10 or len(eo) < 10:
                continue

            # Merge on date
            sf["Date"] = pd.to_datetime(sf["Date"]).dt.date
            eo["Date"] = pd.to_datetime(eo["Date"]).dt.date

            merged = sf.merge(eo, on="Date", suffixes=("_sf", "_eo"))

            if len(merged) < 10:
                continue

            price_col_sf = "Adj. Close_sf" if "Adj. Close_sf" in merged.columns else "Close_sf"
            price_col_eo = "Adj. Close_eo" if "Adj. Close_eo" in merged.columns else "Close_eo"

            if price_col_sf in merged.columns and price_col_eo in merged.columns:
                corr = merged[price_col_sf].corr(merged[price_col_eo])
                if not np.isnan(corr):
                    correlations.append(corr)

        report["tickers_compared"] = len(correlations)
        if correlations:
            report["mean_correlation"] = round(np.mean(correlations), 4)
            report["min_correlation"] = round(np.min(correlations), 4)
            report["max_correlation"] = round(np.max(correlations), 4)
        else:
            report["error"] = "Could not compute price correlations"

        return report

    def compare_coverage(
        self,
        simfin_df: pd.DataFrame,
        eodhd_df: pd.DataFrame,
    ) -> Dict:
        """
        Compare data coverage (date ranges, ticker counts).

        Args:
            simfin_df: SimFin DataFrame
            eodhd_df: EODHD DataFrame

        Returns:
            Coverage comparison report
        """
        report = {
            "simfin": {},
            "eodhd": {},
        }

        # SimFin stats
        if "Ticker" in simfin_df.columns:
            report["simfin"]["n_tickers"] = simfin_df["Ticker"].nunique()
        if "Date" in simfin_df.columns:
            report["simfin"]["date_range"] = (
                str(simfin_df["Date"].min()),
                str(simfin_df["Date"].max()),
            )
        if "Publish Date" in simfin_df.columns:
            report["simfin"]["date_range"] = (
                str(simfin_df["Publish Date"].min()),
                str(simfin_df["Publish Date"].max()),
            )
        report["simfin"]["n_rows"] = len(simfin_df)

        # EODHD stats
        if "Ticker" in eodhd_df.columns:
            report["eodhd"]["n_tickers"] = eodhd_df["Ticker"].nunique()
        if "Date" in eodhd_df.columns:
            report["eodhd"]["date_range"] = (
                str(eodhd_df["Date"].min()),
                str(eodhd_df["Date"].max()),
            )
        if "Publish Date" in eodhd_df.columns:
            report["eodhd"]["date_range"] = (
                str(eodhd_df["Publish Date"].min()),
                str(eodhd_df["Publish Date"].max()),
            )
        report["eodhd"]["n_rows"] = len(eodhd_df)

        return report

    def generate_full_report(
        self,
        simfin_loader,
        eodhd_loader,
    ) -> Dict:
        """
        Generate a comprehensive comparison report.

        Args:
            simfin_loader: SimFinLoader instance
            eodhd_loader: EODHDLoader instance

        Returns:
            Full comparison report dictionary
        """
        report = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "sections": {},
        }

        # Compare fundamentals
        try:
            sf_fund = simfin_loader.load_fundamentals()
            eo_fund = eodhd_loader.load_fundamentals()
            report["sections"]["fundamentals"] = self.compare_fundamentals(
                sf_fund, eo_fund
            )
        except Exception as e:
            report["sections"]["fundamentals"] = {"error": str(e)}

        # Compare prices
        try:
            sf_prices = simfin_loader.load_prices()
            eo_prices = eodhd_loader.load_prices()
            report["sections"]["prices"] = self.compare_prices(
                sf_prices, eo_prices
            )
        except Exception as e:
            report["sections"]["prices"] = {"error": str(e)}

        # Compare coverage
        try:
            report["sections"]["coverage"] = {
                "fundamentals": self.compare_coverage(sf_fund, eo_fund),
                "prices": self.compare_coverage(sf_prices, eo_prices),
            }
        except Exception as e:
            report["sections"]["coverage"] = {"error": str(e)}

        # Recommendation
        report["recommendation"] = self._generate_recommendation(report)

        return report

    def _generate_recommendation(self, report: Dict) -> str:
        """Generate a recommendation based on comparison results."""
        fund_section = report.get("sections", {}).get("fundamentals", {})
        price_section = report.get("sections", {}).get("prices", {})

        issues = []

        # Check fundamental correlations
        metrics = fund_section.get("metric_comparisons", {})
        low_corr_metrics = []
        for metric, data in metrics.items():
            if data.get("status") == "compared":
                if data.get("correlation", 1.0) < 0.9:
                    low_corr_metrics.append(metric)

        if low_corr_metrics:
            issues.append(f"Low correlation for: {', '.join(low_corr_metrics)}")

        # Check price correlations
        mean_price_corr = price_section.get("mean_correlation", 1.0)
        if mean_price_corr < 0.99:
            issues.append(f"Price correlation is {mean_price_corr:.2%}")

        # Check overlap
        overlap_pct = fund_section.get("overlap_pct", 0)
        if overlap_pct < 80:
            issues.append(f"Only {overlap_pct:.1f}% ticker overlap")

        if not issues:
            return "Both data sources appear consistent. Either can be used."
        else:
            return f"Review needed: {'; '.join(issues)}"

    def print_comparison_report(self, report: Dict) -> None:
        """Print a comparison report to console."""
        print("\n" + "=" * 70)
        print("DATA SOURCE COMPARISON REPORT")
        print("=" * 70)

        # Fundamentals
        fund = report.get("sections", {}).get("fundamentals", {})
        print("\nFUNDAMENTALS COMPARISON:")
        print(f"  SimFin tickers: {fund.get('simfin_tickers', 'N/A')}")
        print(f"  EODHD tickers: {fund.get('eodhd_tickers', 'N/A')}")
        print(f"  Overlap: {fund.get('overlapping_tickers', 'N/A')} ({fund.get('overlap_pct', 0):.1f}%)")

        metrics = fund.get("metric_comparisons", {})
        if metrics:
            print("\n  Metric Correlations:")
            for metric, data in metrics.items():
                if data.get("status") == "compared":
                    print(f"    {metric}: r={data.get('correlation', 'N/A')}, "
                          f"MAPE={data.get('mape_pct', 'N/A')}%, "
                          f"Match@10%={data.get('match_rate_10pct', 'N/A')}%")
                else:
                    print(f"    {metric}: {data.get('status', 'unknown')}")

        # Prices
        prices = report.get("sections", {}).get("prices", {})
        print("\nPRICE COMPARISON:")
        print(f"  Tickers compared: {prices.get('tickers_compared', 'N/A')}")
        print(f"  Mean correlation: {prices.get('mean_correlation', 'N/A')}")
        print(f"  Min correlation: {prices.get('min_correlation', 'N/A')}")

        # Recommendation
        print("\nRECOMMENDATION:")
        print(f"  {report.get('recommendation', 'N/A')}")

        print("=" * 70)
