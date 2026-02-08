"""
Unified backtest engine.

Implements strict point-in-time backtesting with fee simulation
and comprehensive performance tracking.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..models.stock_selection_rf import StockSelectionRF
from ..trading.fee_calculator import FeeCalculator
from ..tax.ireland_cgt import IrelandCGTCalculator
from .benchmark import BenchmarkManager
from .metrics import PerformanceMetrics, calculate_metrics
from ..config import (
    BACKTEST_DEFAULTS,
    MARKET_CAP_HIERARCHY,
    OUTLIER_FILTER,
    REBALANCE_TO_TARGET,
    RESULTS_DIR,
)

logger = logging.getLogger(__name__)


@dataclass
class PeriodResult:
    """Result for a single backtest period."""

    date: datetime
    portfolio_return: float
    benchmark_return: float  # Universe benchmark (SimFin)
    sp500_return: float  # S&P 500 benchmark
    n_stocks: int
    selected_stocks: List[str]
    predictions: Dict[str, float]
    actual_returns: Dict[str, float]
    fees_paid: float = 0.0
    turnover: float = 0.0


@dataclass
class BacktestResult:
    """Complete backtest result."""

    periods: List[PeriodResult]
    metrics: PerformanceMetrics  # Portfolio vs Universe benchmark
    metrics_vs_sp500: PerformanceMetrics = None  # Portfolio vs S&P 500
    benchmark_metrics: PerformanceMetrics = None  # Universe benchmark standalone
    sp500_metrics: PerformanceMetrics = None  # S&P 500 standalone

    # Time series
    portfolio_returns: pd.Series = field(default_factory=pd.Series)
    benchmark_returns: pd.Series = field(default_factory=pd.Series)  # Universe
    sp500_returns: pd.Series = field(default_factory=pd.Series)  # S&P 500
    cumulative_returns: pd.Series = field(default_factory=pd.Series)
    benchmark_cumulative: pd.Series = field(default_factory=pd.Series)
    sp500_cumulative: pd.Series = field(default_factory=pd.Series)

    # Summary stats
    total_fees: float = 0.0
    avg_turnover: float = 0.0
    n_periods: int = 0

    def to_dataframe(self) -> pd.DataFrame:
        """Convert period results to DataFrame."""
        data = []
        for p in self.periods:
            data.append({
                "date": p.date,
                "portfolio_return": p.portfolio_return,
                "benchmark_return": p.benchmark_return,
                "sp500_return": p.sp500_return,
                "n_stocks": p.n_stocks,
                "fees": p.fees_paid,
                "turnover": p.turnover,
            })
        return pd.DataFrame(data)


class BacktestEngine:
    """
    Unified backtest engine with strict PIT enforcement.

    Features:
    - Strict point-in-time data alignment
    - Expanding window training
    - Fee simulation
    - Tax impact tracking (optional)
    - Cached benchmark data

    Example:
        model = StockSelectionRF()
        engine = BacktestEngine(model)
        result = engine.run(data, start_date, end_date, n_stocks=15)
    """

    def __init__(
        self,
        model: Optional[StockSelectionRF] = None,
        fee_calculator: Optional[FeeCalculator] = None,
        tax_calculator: Optional[IrelandCGTCalculator] = None,
        benchmark_manager: Optional[BenchmarkManager] = None,
    ):
        """
        Initialize the backtest engine.

        Args:
            model: Stock selection model (creates default if None)
            fee_calculator: For fee simulation
            tax_calculator: For tax impact tracking
            benchmark_manager: For benchmark returns
        """
        self.model = model or StockSelectionRF()
        self.fees = fee_calculator or FeeCalculator()
        self.tax = tax_calculator
        self.benchmark = benchmark_manager or BenchmarkManager()

    def run(
        self,
        data: pd.DataFrame,
        start_date: datetime,
        end_date: datetime,
        rebalance_freq: str = "Q",
        rebalance_month: int = 3,
        n_stocks: int = None,
        initial_capital: float = None,
        min_market_cap: str = None,
        target_col: str = None,  # Auto-select based on rebalance_freq if None
        simulate_fees: bool = True,
        verbose: bool = True,
        benchmark_source: str = "simfin",
        benchmark_weighting: str = "cap_weighted",
    ) -> BacktestResult:
        """
        Run backtest with strict PIT.

        Args:
            data: Panel dataset with features and returns
            start_date: Backtest start date
            end_date: Backtest end date
            rebalance_freq: 'A' for annual, 'Q' for quarterly, 'M' for monthly
            rebalance_month: Month for annual rebalancing (1-12, default 3=March)
            n_stocks: Number of stocks per portfolio
            initial_capital: Starting capital for fee calculation
            min_market_cap: Minimum market cap filter
            target_col: Column with forward returns
            simulate_fees: Whether to simulate trading fees
            verbose: Print progress
            benchmark_source: 'simfin' (default) or 'yfinance'. SimFin uses
                             panel data; yfinance fetches S&P 500.
            benchmark_weighting: 'cap_weighted' (default) or 'equal_weighted'
                                for SimFin benchmark.

        Returns:
            BacktestResult with all metrics and period details
        """
        # Apply defaults
        n_stocks = n_stocks or BACKTEST_DEFAULTS["n_stocks"]
        initial_capital = initial_capital or BACKTEST_DEFAULTS["initial_capital"]
        min_market_cap = min_market_cap or BACKTEST_DEFAULTS["min_market_cap"]

        # Auto-select target column based on rebalance frequency if not specified
        if target_col is None:
            target_col = REBALANCE_TO_TARGET.get(rebalance_freq, "1yr_return")
            logger.info(
                f"Auto-selected target '{target_col}' for {rebalance_freq} rebalancing"
            )

        # Prepare data
        data = self._prepare_data(data, target_col)

        # Filter universe
        data = self._filter_universe(data, min_market_cap)

        # Ensure sector column exists
        if "sector" not in data.columns:
            data = self._infer_sector(data)

        # Generate rebalance dates
        rebalance_dates = self._get_rebalance_dates(
            data, start_date, end_date, rebalance_freq, rebalance_month
        )

        if verbose:
            freq_desc = {
                "A": f"annual (month {rebalance_month})",
                "Y": f"annual (month {rebalance_month})",
                "Q": "quarterly",
                "M": "monthly",
            }.get(rebalance_freq, rebalance_freq)
            logger.info(
                f"Backtest: {len(rebalance_dates)} {freq_desc} periods "
                f"from {start_date.date()} to {end_date.date()}"
            )

        # Always fetch S&P 500 data for comparison
        self.benchmark.fetch_data(start_date, end_date)

        # Store benchmark settings
        self._benchmark_source = benchmark_source
        self._benchmark_weighting = benchmark_weighting
        self._target_col = target_col

        # Run backtest
        periods = []
        prev_holdings = set()
        capital = initial_capital

        for test_date in rebalance_dates:
            if verbose:
                logger.info(f"Rebalancing: {test_date.date()}")

            result = self._run_period(
                data, test_date, target_col, n_stocks, capital, prev_holdings
            )

            if result is not None:
                periods.append(result)
                prev_holdings = set(result.selected_stocks)

                # Update capital
                if simulate_fees:
                    capital = capital * (1 + result.portfolio_return) - result.fees_paid
                else:
                    capital = capital * (1 + result.portfolio_return)

        # Compile results
        return self._compile_results(periods, rebalance_freq)

    def _prepare_data(self, data: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """Prepare data for backtesting."""
        data = data.copy()

        # Ensure datetime
        data["public_date"] = pd.to_datetime(data["public_date"])

        # Snap to month-end
        data["public_date"] = data["public_date"] + pd.offsets.MonthEnd(0)

        # Sort
        data = data.sort_values(["TICKER", "public_date"])

        # Ensure outcome_date exists for PIT filtering
        # Map target column to months forward
        target_to_months = {
            "1mo_return": 1,
            "3mo_return": 3,
            "6mo_return": 6,
            "1yr_return": 12,
        }
        months_forward = target_to_months.get(target_col, 3)

        if "outcome_date" not in data.columns:
            data["outcome_date"] = data["public_date"] + pd.DateOffset(months=months_forward)

        return data

    def _filter_universe(self, data: pd.DataFrame, min_market_cap: str) -> pd.DataFrame:
        """Filter by market cap."""
        if "cap" not in data.columns or min_market_cap == "Nano Cap":
            return data

        try:
            min_idx = MARKET_CAP_HIERARCHY.index(min_market_cap)
            allowed_caps = MARKET_CAP_HIERARCHY[min_idx:]
            data = data[data["cap"].isin(allowed_caps)]
            logger.info(f"Filtered to {min_market_cap}+: {len(data)} rows")
        except ValueError:
            logger.warning(f"Unknown market cap: {min_market_cap}")

        return data

    def _infer_sector(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create pseudo-sectors if sector column missing."""
        logger.info("Creating pseudo-sectors from size/profitability")

        data = data.copy()

        # Create size buckets
        median_cap = data.groupby("TICKER")["MthCap"].transform("median")
        data["size_bucket"] = pd.qcut(
            median_cap, q=3, labels=["Small", "Mid", "Large"], duplicates="drop"
        )

        # Create profitability buckets
        if "roe" in data.columns:
            median_roe = data.groupby("TICKER")["roe"].transform("median")
            data["profit_bucket"] = pd.qcut(
                median_roe, q=3, labels=["Low", "Med", "High"], duplicates="drop"
            )
            data["sector"] = (
                data["size_bucket"].astype(str) + "-" + data["profit_bucket"].astype(str)
            )
        else:
            data["sector"] = data["size_bucket"].astype(str)

        return data

    def _get_rebalance_dates(
        self,
        data: pd.DataFrame,
        start: datetime,
        end: datetime,
        freq: str,
        rebalance_month: int = 3,
    ) -> List[datetime]:
        """
        Get rebalance dates based on frequency.

        Args:
            data: Panel data with public_date column
            start: Start date
            end: End date
            freq: 'A' for annual, 'Q' for quarterly, 'M' for monthly
            rebalance_month: Month for annual rebalancing (1-12, default March)

        Returns:
            List of rebalance dates
        """
        all_dates = sorted(data["public_date"].unique())

        if freq == "A" or freq == "Y":
            # Annual: specific month each year
            dates = [
                d for d in all_dates
                if d.year >= start.year
                and d.year <= end.year
                and d.month == rebalance_month
                and d >= start
                and d <= end
            ]
            logger.info(
                f"Annual rebalancing in month {rebalance_month}: {len(dates)} dates"
            )
        elif freq == "Q":
            # Quarterly: end of Mar, Jun, Sep, Dec
            dates = [
                d for d in all_dates
                if d.year >= start.year
                and d.year <= end.year
                and d.month in [3, 6, 9, 12]
                and d >= start
                and d <= end
            ]
        elif freq == "M":
            # Monthly
            dates = [d for d in all_dates if d >= start and d <= end]
        else:
            raise ValueError(f"Unknown frequency: {freq}. Use 'A', 'Q', or 'M'.")

        return dates

    def _run_period(
        self,
        data: pd.DataFrame,
        test_date: datetime,
        target_col: str,
        n_stocks: int,
        capital: float,
        prev_holdings: set,
    ) -> Optional[PeriodResult]:
        """Run a single backtest period."""

        # Strict PIT: training data has outcome_date < test_date
        train_mask = (data["outcome_date"] < test_date) & (data[target_col].notna())
        test_mask = data["public_date"] == test_date

        train_df = data[train_mask].copy()
        test_df = data[test_mask].copy()

        if len(train_df) < 500 or len(test_df) < n_stocks:
            logger.warning(
                f"Insufficient data for {test_date.date()}: "
                f"train={len(train_df)}, test={len(test_df)}"
            )
            return None

        # Prepare features
        feature_cols = self.model.prepare_features(train_df)

        X_train = train_df[feature_cols]
        y_train = train_df[target_col]
        meta_train = train_df[["sector", "public_date", "TICKER"]]

        X_test = test_df[feature_cols]
        y_test = test_df[target_col]
        y_test_full = y_test.copy()  # Keep full universe for benchmark calc
        meta_test = test_df[["sector", "public_date", "TICKER", "MthCap"]]

        # Train model
        self.model.train(X_train, y_train, meta_train)

        # Handle test data missing values
        X_test_clean = X_test.copy()
        for col in self.model.feature_columns:
            if col not in X_test_clean.columns:
                X_test_clean[col] = 0
        X_test_clean = X_test_clean[self.model.feature_columns]
        X_test_clean = X_test_clean.fillna(self.model.feature_medians_).fillna(0)

        # Cross-sectional outlier filter: remove stocks with too many extreme features
        if OUTLIER_FILTER.get("enabled", True):
            keep_idx = self._filter_cross_sectional_outliers(
                X_test_clean[self.model.feature_columns], test_date,
                z_threshold=OUTLIER_FILTER["z_threshold"],
                max_outlier_pct=OUTLIER_FILTER["max_outlier_pct"],
                min_candidates=OUTLIER_FILTER["min_candidates"],
            )
            if len(keep_idx) < len(X_test_clean):
                n_removed = len(X_test_clean) - len(keep_idx)
                logger.info(
                    f"{test_date.date()}: outlier filter removed {n_removed} "
                    f"stocks ({len(keep_idx)} remaining)"
                )
                X_test_clean = X_test_clean.loc[keep_idx]
                meta_test = meta_test.loc[keep_idx]
                y_test = y_test.loc[keep_idx]

        # Add ROE column for factor weighting (if available in original data)
        if "roe" in test_df.columns and "roe" not in X_test_clean.columns:
            X_test_clean["roe"] = test_df.loc[X_test_clean.index, "roe"].values

        # Select top stocks
        top_stocks = self.model.select_stocks(
            X_test_clean, meta_test, n=n_stocks
        )

        selected_indices = top_stocks.index
        selected_tickers = top_stocks["TICKER"].tolist() if "TICKER" in top_stocks.columns else []

        # Get actual returns — fill NaN (delisted stocks) with delist assumption
        delist_return = BACKTEST_DEFAULTS.get("delist_return", -1.0)
        actual_returns_series = y_test.loc[selected_indices].copy()
        n_nan = actual_returns_series.isna().sum()
        if n_nan > 0:
            nan_tickers = [t for t, idx in zip(selected_tickers, selected_indices)
                           if pd.isna(y_test.loc[idx])]
            logger.warning(
                f"{test_date.date()}: {n_nan} delisted stock(s) assumed "
                f"{delist_return:.0%}: {nan_tickers}"
            )
        actual_returns_series = actual_returns_series.fillna(delist_return)
        portfolio_return = actual_returns_series.mean()

        # Map ticker to actual return
        actual_returns_dict = {}
        for idx, ticker in zip(selected_indices, selected_tickers):
            if idx in y_test.index:
                ret = y_test.loc[idx]
                actual_returns_dict[ticker] = float(ret) if pd.notna(ret) else delist_return

        # Get benchmark returns (use full universe, not outlier-filtered)
        benchmark_return = self._calculate_benchmark_return(
            test_df, y_test_full, target_col
        )
        sp500_return = self._calculate_sp500_return(test_date, target_col)

        # Calculate turnover
        current_holdings = set(selected_tickers)
        if prev_holdings:
            turnover = 1 - len(current_holdings & prev_holdings) / len(current_holdings)
        else:
            turnover = 1.0

        # Estimate fees
        fees_paid = 0.0
        if turnover > 0:
            n_trades = int(n_stocks * turnover * 2)  # buys + sells
            avg_trade_value = capital / n_stocks
            avg_shares = int(avg_trade_value / 100)  # Assume $100 avg price
            fees_paid = n_trades * self.fees.calculate_fee(avg_shares, 100)

        return PeriodResult(
            date=test_date,
            portfolio_return=portfolio_return,
            benchmark_return=benchmark_return,
            sp500_return=sp500_return,
            n_stocks=len(selected_tickers),
            selected_stocks=selected_tickers,
            predictions=dict(zip(selected_tickers, top_stocks["predicted_rank"].tolist())),
            actual_returns=actual_returns_dict,
            fees_paid=fees_paid,
            turnover=turnover,
        )

    def _filter_cross_sectional_outliers(
        self,
        X: pd.DataFrame,
        test_date: datetime,
        z_threshold: float,
        max_outlier_pct: float,
        min_candidates: int,
    ) -> pd.Index:
        """
        Filter stocks with too many cross-sectional outlier features.

        For each stock, compute z-scores across all features. If more than
        max_outlier_pct of features exceed z_threshold σ, exclude that stock.
        Safety valve: never filter below min_candidates.

        Returns:
            Index of stocks to keep.
        """
        means = X.mean()
        stds = X.std().replace(0, np.nan)
        z_scores = (X - means).div(stds)
        outlier_pct = (z_scores.abs() > z_threshold).sum(axis=1) / z_scores.shape[1]
        is_outlier = outlier_pct > max_outlier_pct

        # Safety valve: never filter below min_candidates
        n_outliers = is_outlier.sum()
        if len(X) - n_outliers < min_candidates:
            n_to_exclude = max(0, len(X) - min_candidates)
            if n_to_exclude > 0:
                worst = outlier_pct.nlargest(n_to_exclude)
                is_outlier = X.index.isin(worst.index)
            else:
                is_outlier = pd.Series(False, index=X.index)

        return X.index[~is_outlier]

    def _calculate_benchmark_return(
        self,
        test_df: pd.DataFrame,
        y_test: pd.Series,
        target_col: str,
    ) -> float:
        """
        Calculate universe benchmark return from panel data.

        Always computed from the filtered universe (e.g. Mid Cap+),
        independent of the S&P 500 benchmark.

        Args:
            test_df: Test period data with MthCap column
            y_test: Forward returns for the period
            target_col: Target column name (for holding period)

        Returns:
            Benchmark return as decimal
        """
        benchmark_weighting = getattr(self, "_benchmark_weighting", "cap_weighted")
        valid_returns = y_test.dropna()

        if len(valid_returns) > 0:
            if benchmark_weighting == "cap_weighted" and "MthCap" in test_df.columns:
                weights = test_df.loc[valid_returns.index, "MthCap"].copy()
                weights = weights.fillna(0)
                # Clip outlier market caps at 99.5th percentile to prevent
                # bad data (e.g. $100T+ entries) from dominating the benchmark
                cap_clip = weights.quantile(0.995)
                if cap_clip > 0:
                    weights = weights.clip(upper=cap_clip)
                total_weight = weights.sum()

                if total_weight > 0:
                    benchmark_return = (valid_returns * weights).sum() / total_weight
                    logger.debug(
                        f"Universe cap-weighted benchmark: {benchmark_return:.4f} "
                        f"({len(valid_returns)} stocks)"
                    )
                    return benchmark_return

            # Equal-weighted fallback
            benchmark_return = valid_returns.mean()
            logger.debug(
                f"Universe equal-weighted benchmark: {benchmark_return:.4f} "
                f"({len(valid_returns)} stocks)"
            )
            return benchmark_return

        logger.warning("No universe benchmark data available, returning 0")
        return 0.0

    def _calculate_sp500_return(
        self,
        test_date: datetime,
        target_col: str,
    ) -> float:
        """
        Calculate S&P 500 return for the holding period.

        Args:
            test_date: Start date of the period
            target_col: Target column name (determines holding period)

        Returns:
            S&P 500 return as decimal, or 0.0 if unavailable
        """
        if target_col == "1yr_return":
            holding_months = 12
        else:
            holding_months = 3

        end_date = test_date + pd.DateOffset(months=holding_months)
        sp500_return = self.benchmark.get_return(test_date, end_date)

        if sp500_return is not None:
            logger.debug(f"S&P 500 return: {sp500_return:.4f}")
            return sp500_return

        logger.warning(f"S&P 500 data unavailable for {test_date.date()}")
        return 0.0

    def _compile_results(
        self, periods: List[PeriodResult], freq: str
    ) -> BacktestResult:
        """Compile period results into final backtest result."""
        if not periods:
            empty_metrics = calculate_metrics(pd.Series(), periods_per_year=4)
            return BacktestResult(
                periods=[],
                metrics=empty_metrics,
                metrics_vs_sp500=empty_metrics,
                benchmark_metrics=empty_metrics,
                sp500_metrics=empty_metrics,
            )

        # Extract returns series
        dates = [p.date for p in periods]
        port_returns = pd.Series(
            [p.portfolio_return for p in periods], index=dates
        )
        bench_returns = pd.Series(
            [p.benchmark_return for p in periods], index=dates
        )
        sp500_returns = pd.Series(
            [p.sp500_return for p in periods], index=dates
        )

        # Calculate cumulative
        cum_port = (1 + port_returns).cumprod()
        cum_bench = (1 + bench_returns).cumprod()
        cum_sp500 = (1 + sp500_returns).cumprod()

        # Calculate metrics - periods_per_year based on frequency
        if freq in ("A", "Y"):
            periods_per_year = 1
        elif freq == "Q":
            periods_per_year = 4
        else:  # Monthly
            periods_per_year = 12

        metrics = calculate_metrics(
            port_returns, bench_returns, periods_per_year=periods_per_year
        )
        metrics_vs_sp500 = calculate_metrics(
            port_returns, sp500_returns, periods_per_year=periods_per_year
        )
        bench_metrics = calculate_metrics(
            bench_returns, periods_per_year=periods_per_year
        )
        sp500_metrics = calculate_metrics(
            sp500_returns, periods_per_year=periods_per_year
        )

        # Summary stats
        total_fees = sum(p.fees_paid for p in periods)
        avg_turnover = np.mean([p.turnover for p in periods])

        return BacktestResult(
            periods=periods,
            metrics=metrics,
            metrics_vs_sp500=metrics_vs_sp500,
            benchmark_metrics=bench_metrics,
            sp500_metrics=sp500_metrics,
            portfolio_returns=port_returns,
            benchmark_returns=bench_returns,
            sp500_returns=sp500_returns,
            cumulative_returns=cum_port,
            benchmark_cumulative=cum_bench,
            sp500_cumulative=cum_sp500,
            total_fees=total_fees,
            avg_turnover=avg_turnover,
            n_periods=len(periods),
        )

    def print_summary(self, result: BacktestResult) -> None:
        """Print backtest summary."""
        print("\n" + "=" * 70)
        print("BACKTEST SUMMARY")
        print("=" * 70)

        print(f"\nPeriods: {result.n_periods}")
        print(f"Total Fees: ${result.total_fees:.2f}")
        print(f"Avg Turnover: {result.avg_turnover * 100:.1f}%")

        print("\n--- PORTFOLIO ---")
        m = result.metrics
        print(f"Total Return:      {m.total_return * 100:>8.2f}%")
        print(f"Annualized Return: {m.annualized_return * 100:>8.2f}%")
        print(f"Sharpe Ratio:      {m.sharpe_ratio:>8.2f}")
        print(f"Max Drawdown:      {m.max_drawdown * 100:>8.2f}%")

        print("\n--- UNIVERSE BENCHMARK (Mid Cap+) ---")
        b = result.benchmark_metrics
        print(f"Total Return:      {b.total_return * 100:>8.2f}%")
        print(f"Annualized Return: {b.annualized_return * 100:>8.2f}%")
        print(f"Sharpe Ratio:      {b.sharpe_ratio:>8.2f}")
        print(f"Max Drawdown:      {b.max_drawdown * 100:>8.2f}%")

        print("\n--- S&P 500 ---")
        s = result.sp500_metrics
        if s is not None:
            print(f"Total Return:      {s.total_return * 100:>8.2f}%")
            print(f"Annualized Return: {s.annualized_return * 100:>8.2f}%")
            print(f"Sharpe Ratio:      {s.sharpe_ratio:>8.2f}")
            print(f"Max Drawdown:      {s.max_drawdown * 100:>8.2f}%")
        else:
            print("(Data unavailable)")

        print("\n--- VS UNIVERSE BENCHMARK ---")
        print(f"Alpha:             {m.alpha * 100:>8.2f}%")
        print(f"Beta:              {m.beta:>8.2f}")
        print(f"Excess Return:     {m.excess_return * 100:>8.2f}%")
        print(f"Win Rate:          {m.win_rate * 100:>8.1f}%")

        print("\n--- VS S&P 500 ---")
        sp = result.metrics_vs_sp500
        if sp is not None:
            print(f"Alpha:             {sp.alpha * 100:>8.2f}%")
            print(f"Beta:              {sp.beta:>8.2f}")
            print(f"Excess Return:     {sp.excess_return * 100:>8.2f}%")
            print(f"Win Rate:          {sp.win_rate * 100:>8.1f}%")
        else:
            print("(Data unavailable)")

        # Year-by-year comparison
        self._print_yearly_comparison(result)

        print("=" * 70)

    def _print_yearly_comparison(self, result: BacktestResult) -> None:
        """Print year-by-year return comparison table."""
        if not result.periods:
            return

        # Aggregate period returns into annual returns by compounding
        yearly = {}
        for p in result.periods:
            year = p.date.year
            if year not in yearly:
                yearly[year] = {"port": 1.0, "univ": 1.0, "sp500": 1.0}
            yearly[year]["port"] *= (1 + p.portfolio_return)
            yearly[year]["univ"] *= (1 + p.benchmark_return)
            yearly[year]["sp500"] *= (1 + p.sp500_return)

        print("\n--- YEAR-BY-YEAR COMPARISON ---")
        print(f"{'Year':>6}  {'Portfolio':>10}  {'Universe':>10}  {'S&P 500':>10}  {'vs Univ':>10}  {'vs S&P':>10}")
        print("-" * 70)

        for year in sorted(yearly):
            port = (yearly[year]["port"] - 1) * 100
            univ = (yearly[year]["univ"] - 1) * 100
            sp = (yearly[year]["sp500"] - 1) * 100
            vs_univ = port - univ
            vs_sp = port - sp

            print(
                f"{year:>6}  {port:>9.1f}%  {univ:>9.1f}%  {sp:>9.1f}%"
                f"  {vs_univ:>+9.1f}%  {vs_sp:>+9.1f}%"
            )
