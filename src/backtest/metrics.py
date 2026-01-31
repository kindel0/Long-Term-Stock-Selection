"""
Performance metrics calculation.

Provides comprehensive performance metrics for backtest analysis.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """
    Comprehensive performance metrics for a strategy.

    Contains all standard metrics for evaluating strategy performance.
    """

    # Returns
    total_return: float
    annualized_return: float
    avg_period_return: float
    best_period: float
    worst_period: float

    # Risk
    volatility: float
    downside_deviation: float
    max_drawdown: float
    max_drawdown_duration: int  # in periods

    # Risk-adjusted
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    information_ratio: float

    # Benchmark comparison
    alpha: float
    beta: float
    tracking_error: float
    excess_return: float
    win_rate: float  # fraction of periods beating benchmark

    # Other
    n_periods: int
    periods_per_year: int  # 4 for quarterly, 12 for monthly

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "total_return_pct": self.total_return * 100,
            "annualized_return_pct": self.annualized_return * 100,
            "avg_period_return_pct": self.avg_period_return * 100,
            "best_period_pct": self.best_period * 100,
            "worst_period_pct": self.worst_period * 100,
            "volatility_pct": self.volatility * 100,
            "max_drawdown_pct": self.max_drawdown * 100,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "calmar_ratio": self.calmar_ratio,
            "information_ratio": self.information_ratio,
            "alpha_pct": self.alpha * 100,
            "beta": self.beta,
            "excess_return_pct": self.excess_return * 100,
            "win_rate_pct": self.win_rate * 100,
            "n_periods": self.n_periods,
        }


def calculate_metrics(
    returns: pd.Series,
    benchmark_returns: Optional[pd.Series] = None,
    risk_free_rate: float = 0.02,
    periods_per_year: int = 4,
) -> PerformanceMetrics:
    """
    Calculate comprehensive performance metrics.

    Args:
        returns: Series of period returns
        benchmark_returns: Series of benchmark returns (same index)
        risk_free_rate: Annual risk-free rate (default 2%)
        periods_per_year: Number of periods per year (4 for quarterly)

    Returns:
        PerformanceMetrics object
    """
    returns = returns.dropna()
    n_periods = len(returns)

    if n_periods == 0:
        return _empty_metrics(periods_per_year)

    # Basic return metrics
    total_return = (1 + returns).prod() - 1
    annualized_return = (1 + total_return) ** (periods_per_year / n_periods) - 1
    avg_period_return = returns.mean()
    best_period = returns.max()
    worst_period = returns.min()

    # Volatility
    volatility = returns.std() * np.sqrt(periods_per_year)

    # Downside deviation
    negative_returns = returns[returns < 0]
    downside_deviation = negative_returns.std() * np.sqrt(periods_per_year) if len(negative_returns) > 0 else 0

    # Drawdown
    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.expanding().max()
    drawdown = (cumulative - rolling_max) / rolling_max
    max_drawdown = abs(drawdown.min())

    # Max drawdown duration
    is_underwater = drawdown < 0
    dd_groups = (is_underwater != is_underwater.shift()).cumsum()
    dd_durations = is_underwater.groupby(dd_groups).sum()
    max_dd_duration = int(dd_durations.max()) if len(dd_durations) > 0 else 0

    # Risk-adjusted metrics
    rf_per_period = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
    excess_returns = returns - rf_per_period

    sharpe_ratio = 0.0
    if volatility > 0:
        sharpe_ratio = (annualized_return - risk_free_rate) / volatility

    sortino_ratio = 0.0
    if downside_deviation > 0:
        sortino_ratio = (annualized_return - risk_free_rate) / downside_deviation

    calmar_ratio = 0.0
    if max_drawdown > 0:
        calmar_ratio = annualized_return / max_drawdown

    # Benchmark comparison
    alpha = 0.0
    beta = 0.0
    tracking_error = 0.0
    information_ratio = 0.0
    excess_return = 0.0
    win_rate = 0.0

    if benchmark_returns is not None and len(benchmark_returns) > 0:
        # Align returns
        aligned = pd.DataFrame({
            "strategy": returns,
            "benchmark": benchmark_returns
        }).dropna()

        if len(aligned) > 1:
            strat = aligned["strategy"]
            bench = aligned["benchmark"]

            # Beta and alpha
            cov = strat.cov(bench)
            var_bench = bench.var()
            beta = cov / var_bench if var_bench > 0 else 0

            bench_total = (1 + bench).prod() - 1
            bench_ann = (1 + bench_total) ** (periods_per_year / len(bench)) - 1
            alpha = annualized_return - risk_free_rate - beta * (bench_ann - risk_free_rate)

            # Tracking error
            tracking_error = (strat - bench).std() * np.sqrt(periods_per_year)

            # Information ratio
            if tracking_error > 0:
                excess_over_bench = (strat - bench).mean() * periods_per_year
                information_ratio = excess_over_bench / tracking_error

            # Excess return
            excess_return = annualized_return - bench_ann

            # Win rate
            win_rate = (strat > bench).mean()

    return PerformanceMetrics(
        total_return=total_return,
        annualized_return=annualized_return,
        avg_period_return=avg_period_return,
        best_period=best_period,
        worst_period=worst_period,
        volatility=volatility,
        downside_deviation=downside_deviation,
        max_drawdown=max_drawdown,
        max_drawdown_duration=max_dd_duration,
        sharpe_ratio=sharpe_ratio,
        sortino_ratio=sortino_ratio,
        calmar_ratio=calmar_ratio,
        information_ratio=information_ratio,
        alpha=alpha,
        beta=beta,
        tracking_error=tracking_error,
        excess_return=excess_return,
        win_rate=win_rate,
        n_periods=n_periods,
        periods_per_year=periods_per_year,
    )


def _empty_metrics(periods_per_year: int) -> PerformanceMetrics:
    """Return empty metrics when no data available."""
    return PerformanceMetrics(
        total_return=0,
        annualized_return=0,
        avg_period_return=0,
        best_period=0,
        worst_period=0,
        volatility=0,
        downside_deviation=0,
        max_drawdown=0,
        max_drawdown_duration=0,
        sharpe_ratio=0,
        sortino_ratio=0,
        calmar_ratio=0,
        information_ratio=0,
        alpha=0,
        beta=0,
        tracking_error=0,
        excess_return=0,
        win_rate=0,
        n_periods=0,
        periods_per_year=periods_per_year,
    )


def print_metrics(metrics: PerformanceMetrics, name: str = "Strategy") -> None:
    """Print metrics in a formatted table."""
    print(f"\n{'=' * 50}")
    print(f"PERFORMANCE METRICS: {name}")
    print("=" * 50)

    print("\nRETURNS:")
    print(f"  Total Return:      {metrics.total_return * 100:>8.2f}%")
    print(f"  Annualized Return: {metrics.annualized_return * 100:>8.2f}%")
    print(f"  Avg Period Return: {metrics.avg_period_return * 100:>8.2f}%")
    print(f"  Best Period:       {metrics.best_period * 100:>8.2f}%")
    print(f"  Worst Period:      {metrics.worst_period * 100:>8.2f}%")

    print("\nRISK:")
    print(f"  Volatility:        {metrics.volatility * 100:>8.2f}%")
    print(f"  Max Drawdown:      {metrics.max_drawdown * 100:>8.2f}%")
    print(f"  DD Duration:       {metrics.max_drawdown_duration:>8} periods")

    print("\nRISK-ADJUSTED:")
    print(f"  Sharpe Ratio:      {metrics.sharpe_ratio:>8.2f}")
    print(f"  Sortino Ratio:     {metrics.sortino_ratio:>8.2f}")
    print(f"  Calmar Ratio:      {metrics.calmar_ratio:>8.2f}")

    print("\nBENCHMARK COMPARISON:")
    print(f"  Alpha:             {metrics.alpha * 100:>8.2f}%")
    print(f"  Beta:              {metrics.beta:>8.2f}")
    print(f"  Info Ratio:        {metrics.information_ratio:>8.2f}")
    print(f"  Excess Return:     {metrics.excess_return * 100:>8.2f}%")
    print(f"  Win Rate:          {metrics.win_rate * 100:>8.1f}%")

    print(f"\nPeriods: {metrics.n_periods} ({metrics.periods_per_year}/year)")
    print("=" * 50)


def compare_metrics(
    metrics_list: List[PerformanceMetrics],
    names: List[str],
) -> pd.DataFrame:
    """
    Compare metrics across multiple strategies.

    Args:
        metrics_list: List of PerformanceMetrics
        names: Strategy names

    Returns:
        DataFrame comparing key metrics
    """
    data = []
    for metrics, name in zip(metrics_list, names):
        data.append({
            "Strategy": name,
            "Total Return": f"{metrics.total_return * 100:.1f}%",
            "Ann. Return": f"{metrics.annualized_return * 100:.1f}%",
            "Volatility": f"{metrics.volatility * 100:.1f}%",
            "Sharpe": f"{metrics.sharpe_ratio:.2f}",
            "Max DD": f"{metrics.max_drawdown * 100:.1f}%",
            "Win Rate": f"{metrics.win_rate * 100:.0f}%",
        })

    return pd.DataFrame(data)
