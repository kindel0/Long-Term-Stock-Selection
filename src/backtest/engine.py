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
    RESULTS_DIR,
)

logger = logging.getLogger(__name__)


@dataclass
class PeriodResult:
    """Result for a single backtest period."""

    date: datetime
    portfolio_return: float
    benchmark_return: float
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
    metrics: PerformanceMetrics
    benchmark_metrics: PerformanceMetrics

    # Time series
    portfolio_returns: pd.Series = field(default_factory=pd.Series)
    benchmark_returns: pd.Series = field(default_factory=pd.Series)
    cumulative_returns: pd.Series = field(default_factory=pd.Series)
    benchmark_cumulative: pd.Series = field(default_factory=pd.Series)

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
        n_stocks: int = None,
        initial_capital: float = None,
        min_market_cap: str = None,
        target_col: str = "3mo_return",
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
            rebalance_freq: 'Q' for quarterly, 'M' for monthly
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

        # Prepare data
        data = self._prepare_data(data, target_col)

        # Filter universe
        data = self._filter_universe(data, min_market_cap)

        # Ensure sector column exists
        if "sector" not in data.columns:
            data = self._infer_sector(data)

        # Generate rebalance dates
        rebalance_dates = self._get_rebalance_dates(
            data, start_date, end_date, rebalance_freq
        )

        if verbose:
            logger.info(f"Backtest: {len(rebalance_dates)} periods from {start_date.date()} to {end_date.date()}")

        # Fetch yfinance benchmark data only if needed as fallback
        if benchmark_source == "yfinance":
            self.benchmark.fetch_data(start_date, end_date)

        # Store benchmark settings
        self._benchmark_source = benchmark_source
        self._benchmark_weighting = benchmark_weighting

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
        if "outcome_date" not in data.columns:
            if target_col == "3mo_return":
                data["outcome_date"] = data["public_date"] + pd.DateOffset(months=3)
            elif target_col == "1yr_return":
                data["outcome_date"] = data["public_date"] + pd.DateOffset(months=12)
            else:
                # Default to 3 months
                data["outcome_date"] = data["public_date"] + pd.DateOffset(months=3)

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
    ) -> List[datetime]:
        """Get rebalance dates based on frequency."""
        all_dates = sorted(data["public_date"].unique())

        if freq == "Q":
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
            raise ValueError(f"Unknown frequency: {freq}")

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

        # Count stocks with valid forward returns
        n_valid_returns = test_df[target_col].notna().sum()

        if len(train_df) < 500 or len(test_df) < n_stocks:
            logger.warning(
                f"Insufficient data for {test_date.date()}: "
                f"train={len(train_df)}, test={len(test_df)}"
            )
            return None

        if n_valid_returns < n_stocks:
            logger.warning(
                f"Insufficient forward returns for {test_date.date()}: "
                f"only {n_valid_returns} stocks have valid {target_col}"
            )
            return None

        # Prepare features
        feature_cols = self.model.prepare_features(train_df)

        X_train = train_df[feature_cols]
        y_train = train_df[target_col]
        meta_train = train_df[["sector", "public_date", "TICKER"]]

        X_test = test_df[feature_cols]
        y_test = test_df[target_col]
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

        # Select top stocks
        top_stocks = self.model.select_stocks(
            X_test_clean, meta_test, n=n_stocks
        )

        selected_indices = top_stocks.index
        selected_tickers = top_stocks["TICKER"].tolist() if "TICKER" in top_stocks.columns else []

        # Get actual returns
        actual_returns = y_test.loc[selected_indices].dropna()
        portfolio_return = actual_returns.mean() if len(actual_returns) > 0 else 0.0

        # Get benchmark return - prefer SimFin data
        benchmark_return = self._calculate_benchmark_return(
            test_df, y_test, target_col
        )

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
            n_stocks=len(actual_returns),
            selected_stocks=selected_tickers,
            predictions=dict(zip(selected_tickers, top_stocks["predicted_rank"].tolist())),
            actual_returns=actual_returns.to_dict(),
            fees_paid=fees_paid,
            turnover=turnover,
        )

    def _calculate_benchmark_return(
        self,
        test_df: pd.DataFrame,
        y_test: pd.Series,
        target_col: str,
    ) -> float:
        """
        Calculate benchmark return from SimFin data or yfinance fallback.

        Priority:
        1. SimFin data (cap-weighted or equal-weighted based on settings)
        2. yfinance S&P 500 (only if benchmark_source='yfinance' or SimFin fails)

        Args:
            test_df: Test period data with MthCap column
            y_test: Forward returns for the period
            target_col: Target column name (for holding period)

        Returns:
            Benchmark return as decimal
        """
        benchmark_source = getattr(self, "_benchmark_source", "simfin")
        benchmark_weighting = getattr(self, "_benchmark_weighting", "cap_weighted")

        # Try SimFin-based benchmark first (unless yfinance explicitly requested)
        if benchmark_source == "simfin":
            valid_returns = y_test.dropna()

            if len(valid_returns) > 0:
                if benchmark_weighting == "cap_weighted" and "MthCap" in test_df.columns:
                    # Cap-weighted average
                    weights = test_df.loc[valid_returns.index, "MthCap"]
                    weights = weights.fillna(0)
                    total_weight = weights.sum()

                    if total_weight > 0:
                        benchmark_return = (valid_returns * weights).sum() / total_weight
                        logger.debug(
                            f"SimFin cap-weighted benchmark: {benchmark_return:.4f} "
                            f"({len(valid_returns)} stocks)"
                        )
                        return benchmark_return

                # Equal-weighted fallback
                benchmark_return = valid_returns.mean()
                logger.debug(
                    f"SimFin equal-weighted benchmark: {benchmark_return:.4f} "
                    f"({len(valid_returns)} stocks)"
                )
                return benchmark_return

        # yfinance fallback (S&P 500)
        if target_col == "1yr_return":
            holding_months = 12
        else:
            holding_months = 3

        test_date = test_df["public_date"].iloc[0] if len(test_df) > 0 else None
        if test_date is not None:
            end_date = test_date + pd.DateOffset(months=holding_months)
            benchmark_return = self.benchmark.get_return(test_date, end_date)

            if benchmark_return is not None:
                logger.debug(f"yfinance S&P 500 benchmark: {benchmark_return:.4f}")
                return benchmark_return

        # Final fallback: equal-weighted SimFin
        valid_returns = y_test.dropna()
        if len(valid_returns) > 0:
            benchmark_return = valid_returns.mean()
            logger.warning(
                f"Using equal-weighted fallback benchmark: {benchmark_return:.4f}"
            )
            return benchmark_return

        logger.warning("No benchmark data available, returning 0")
        return 0.0

    def _compile_results(
        self, periods: List[PeriodResult], freq: str
    ) -> BacktestResult:
        """Compile period results into final backtest result."""
        if not periods:
            return BacktestResult(
                periods=[],
                metrics=calculate_metrics(pd.Series(), periods_per_year=4),
                benchmark_metrics=calculate_metrics(pd.Series(), periods_per_year=4),
            )

        # Extract returns series
        dates = [p.date for p in periods]
        port_returns = pd.Series(
            [p.portfolio_return for p in periods], index=dates
        )
        bench_returns = pd.Series(
            [p.benchmark_return for p in periods], index=dates
        )

        # Calculate cumulative
        cum_port = (1 + port_returns).cumprod()
        cum_bench = (1 + bench_returns).cumprod()

        # Calculate metrics
        periods_per_year = 4 if freq == "Q" else 12
        metrics = calculate_metrics(
            port_returns, bench_returns, periods_per_year=periods_per_year
        )
        bench_metrics = calculate_metrics(
            bench_returns, periods_per_year=periods_per_year
        )

        # Summary stats
        total_fees = sum(p.fees_paid for p in periods)
        avg_turnover = np.mean([p.turnover for p in periods])

        return BacktestResult(
            periods=periods,
            metrics=metrics,
            benchmark_metrics=bench_metrics,
            portfolio_returns=port_returns,
            benchmark_returns=bench_returns,
            cumulative_returns=cum_port,
            benchmark_cumulative=cum_bench,
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

        print("\n--- BENCHMARK ---")
        b = result.benchmark_metrics
        print(f"Total Return:      {b.total_return * 100:>8.2f}%")
        print(f"Annualized Return: {b.annualized_return * 100:>8.2f}%")

        print("\n--- VS BENCHMARK ---")
        print(f"Alpha:             {m.alpha * 100:>8.2f}%")
        print(f"Beta:              {m.beta:>8.2f}")
        print(f"Excess Return:     {m.excess_return * 100:>8.2f}%")
        print(f"Win Rate:          {m.win_rate * 100:>8.1f}%")

        print("=" * 70)
