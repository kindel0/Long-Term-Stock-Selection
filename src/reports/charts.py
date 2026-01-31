"""
Chart generation for backtest reports.

Creates visualization charts for backtest analysis.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..config import FIGURES_DIR

logger = logging.getLogger(__name__)


class ChartGenerator:
    """
    Generates charts for backtest analysis.

    Uses matplotlib for chart generation. Charts can be saved
    to disk or displayed interactively.

    Example:
        charts = ChartGenerator()
        charts.cumulative_returns(result, save_path="results/cumulative.png")
    """

    def __init__(self, output_dir: Optional[Path] = None, style: str = "seaborn-v0_8-whitegrid"):
        """
        Initialize the chart generator.

        Args:
            output_dir: Directory for saving charts
            style: Matplotlib style to use
        """
        self.output_dir = Path(output_dir) if output_dir else FIGURES_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.style = style

    def _setup_style(self):
        """Apply consistent styling."""
        import matplotlib.pyplot as plt

        try:
            plt.style.use(self.style)
        except Exception:
            plt.style.use("seaborn-v0_8-whitegrid")

    def cumulative_returns(
        self,
        portfolio_cumulative: pd.Series,
        benchmark_cumulative: pd.Series,
        title: str = "Cumulative Returns",
        save_path: Optional[str] = None,
        show: bool = True,
    ) -> None:
        """
        Plot cumulative returns comparison.

        Args:
            portfolio_cumulative: Portfolio cumulative returns
            benchmark_cumulative: Benchmark cumulative returns
            title: Chart title
            save_path: Path to save (relative to output_dir)
            show: Whether to display the chart
        """
        import matplotlib.pyplot as plt

        self._setup_style()
        fig, ax = plt.subplots(figsize=(12, 6))

        ax.plot(
            portfolio_cumulative.index,
            portfolio_cumulative.values,
            label="Portfolio",
            linewidth=2,
            color="#2E86AB",
        )
        ax.plot(
            benchmark_cumulative.index,
            benchmark_cumulative.values,
            label="S&P 500",
            linewidth=2,
            linestyle="--",
            color="#A23B72",
        )

        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel("Growth of $1", fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        # Add final values annotation
        port_final = portfolio_cumulative.iloc[-1]
        bench_final = benchmark_cumulative.iloc[-1]
        ax.annotate(
            f"Portfolio: ${port_final:.2f}",
            xy=(0.02, 0.95),
            xycoords="axes fraction",
            fontsize=10,
        )
        ax.annotate(
            f"Benchmark: ${bench_final:.2f}",
            xy=(0.02, 0.90),
            xycoords="axes fraction",
            fontsize=10,
        )

        plt.tight_layout()

        if save_path:
            filepath = self.output_dir / save_path
            plt.savefig(filepath, dpi=300, bbox_inches="tight")
            logger.info(f"Saved chart: {filepath}")

        if show:
            plt.show()
        else:
            plt.close()

    def drawdown(
        self,
        returns: pd.Series,
        title: str = "Portfolio Drawdown",
        save_path: Optional[str] = None,
        show: bool = True,
    ) -> None:
        """
        Plot drawdown chart.

        Args:
            returns: Period returns
            title: Chart title
            save_path: Path to save
            show: Whether to display
        """
        import matplotlib.pyplot as plt

        self._setup_style()

        # Calculate drawdown
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max * 100

        fig, ax = plt.subplots(figsize=(12, 4))

        ax.fill_between(
            drawdown.index,
            0,
            drawdown.values,
            color="#E74C3C",
            alpha=0.5,
        )
        ax.plot(drawdown.index, drawdown.values, color="#C0392B", linewidth=1)

        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel("Drawdown (%)", fontsize=12)
        ax.grid(True, alpha=0.3)

        # Annotate max drawdown
        max_dd = drawdown.min()
        max_dd_date = drawdown.idxmin()
        ax.annotate(
            f"Max: {max_dd:.1f}%",
            xy=(max_dd_date, max_dd),
            xytext=(10, -20),
            textcoords="offset points",
            fontsize=9,
            arrowprops=dict(arrowstyle="->", color="gray"),
        )

        plt.tight_layout()

        if save_path:
            filepath = self.output_dir / save_path
            plt.savefig(filepath, dpi=300, bbox_inches="tight")
            logger.info(f"Saved chart: {filepath}")

        if show:
            plt.show()
        else:
            plt.close()

    def annual_returns(
        self,
        returns: pd.Series,
        benchmark_returns: Optional[pd.Series] = None,
        title: str = "Annual Returns",
        save_path: Optional[str] = None,
        show: bool = True,
    ) -> None:
        """
        Plot annual returns bar chart.

        Args:
            returns: Period returns
            benchmark_returns: Optional benchmark returns
            title: Chart title
            save_path: Path to save
            show: Whether to display
        """
        import matplotlib.pyplot as plt

        self._setup_style()

        # Calculate annual returns
        annual = (1 + returns).groupby(returns.index.year).prod() - 1

        fig, ax = plt.subplots(figsize=(10, 6))

        x = np.arange(len(annual))
        width = 0.35

        ax.bar(
            x - width / 2 if benchmark_returns is not None else x,
            annual.values * 100,
            width,
            label="Portfolio",
            color="#2E86AB",
        )

        if benchmark_returns is not None:
            bench_annual = (1 + benchmark_returns).groupby(benchmark_returns.index.year).prod() - 1
            ax.bar(
                x + width / 2,
                bench_annual.values * 100,
                width,
                label="Benchmark",
                color="#A23B72",
            )

        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("Year", fontsize=12)
        ax.set_ylabel("Return (%)", fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(annual.index)
        ax.legend(fontsize=10)
        ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()

        if save_path:
            filepath = self.output_dir / save_path
            plt.savefig(filepath, dpi=300, bbox_inches="tight")
            logger.info(f"Saved chart: {filepath}")

        if show:
            plt.show()
        else:
            plt.close()

    def feature_importance(
        self,
        importances: pd.DataFrame,
        top_n: int = 15,
        title: str = "Feature Importance",
        save_path: Optional[str] = None,
        show: bool = True,
    ) -> None:
        """
        Plot feature importance.

        Args:
            importances: DataFrame with 'feature' and 'importance' columns
            top_n: Number of top features to show
            title: Chart title
            save_path: Path to save
            show: Whether to display
        """
        import matplotlib.pyplot as plt

        self._setup_style()

        top = importances.nlargest(top_n, "importance")

        fig, ax = plt.subplots(figsize=(10, 8))

        colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(top)))

        ax.barh(
            range(len(top)),
            top["importance"].values,
            color=colors[::-1],
        )
        ax.set_yticks(range(len(top)))
        ax.set_yticklabels(top["feature"].values)
        ax.invert_yaxis()

        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("Importance", fontsize=12)
        ax.grid(True, alpha=0.3, axis="x")

        plt.tight_layout()

        if save_path:
            filepath = self.output_dir / save_path
            plt.savefig(filepath, dpi=300, bbox_inches="tight")
            logger.info(f"Saved chart: {filepath}")

        if show:
            plt.show()
        else:
            plt.close()

    def rolling_metrics(
        self,
        returns: pd.Series,
        window: int = 4,
        title: str = "Rolling Performance",
        save_path: Optional[str] = None,
        show: bool = True,
    ) -> None:
        """
        Plot rolling performance metrics.

        Args:
            returns: Period returns
            window: Rolling window size
            title: Chart title
            save_path: Path to save
            show: Whether to display
        """
        import matplotlib.pyplot as plt

        self._setup_style()

        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        # Rolling return
        rolling_ret = returns.rolling(window).mean() * window * 100
        axes[0].plot(rolling_ret.index, rolling_ret.values, color="#2E86AB", linewidth=2)
        axes[0].axhline(y=0, color="black", linestyle="--", linewidth=0.5)
        axes[0].set_ylabel(f"Rolling {window}-Period Return (%)", fontsize=11)
        axes[0].set_title(title, fontsize=14, fontweight="bold")
        axes[0].grid(True, alpha=0.3)

        # Rolling volatility
        rolling_vol = returns.rolling(window).std() * np.sqrt(4) * 100
        axes[1].plot(rolling_vol.index, rolling_vol.values, color="#E74C3C", linewidth=2)
        axes[1].set_ylabel("Rolling Volatility (%)", fontsize=11)
        axes[1].set_xlabel("Date", fontsize=12)
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            filepath = self.output_dir / save_path
            plt.savefig(filepath, dpi=300, bbox_inches="tight")
            logger.info(f"Saved chart: {filepath}")

        if show:
            plt.show()
        else:
            plt.close()

    def sector_allocation(
        self,
        allocations: Dict[str, float],
        title: str = "Sector Allocation",
        save_path: Optional[str] = None,
        show: bool = True,
    ) -> None:
        """
        Plot sector allocation pie chart.

        Args:
            allocations: Dict of sector -> weight
            title: Chart title
            save_path: Path to save
            show: Whether to display
        """
        import matplotlib.pyplot as plt

        self._setup_style()

        fig, ax = plt.subplots(figsize=(10, 8))

        sectors = list(allocations.keys())
        weights = list(allocations.values())

        colors = plt.cm.Set3(np.linspace(0, 1, len(sectors)))

        wedges, texts, autotexts = ax.pie(
            weights,
            labels=sectors,
            autopct="%1.1f%%",
            colors=colors,
            startangle=90,
        )

        ax.set_title(title, fontsize=14, fontweight="bold")

        plt.tight_layout()

        if save_path:
            filepath = self.output_dir / save_path
            plt.savefig(filepath, dpi=300, bbox_inches="tight")
            logger.info(f"Saved chart: {filepath}")

        if show:
            plt.show()
        else:
            plt.close()

    def generate_all(
        self,
        result,  # BacktestResult
        prefix: str = "",
        show: bool = False,
    ) -> List[Path]:
        """
        Generate all standard charts.

        Args:
            result: BacktestResult object
            prefix: Prefix for filenames
            show: Whether to display charts

        Returns:
            List of saved file paths
        """
        paths = []

        # Cumulative returns
        self.cumulative_returns(
            result.cumulative_returns,
            result.benchmark_cumulative,
            save_path=f"{prefix}cumulative_returns.png",
            show=show,
        )
        paths.append(self.output_dir / f"{prefix}cumulative_returns.png")

        # Drawdown
        self.drawdown(
            result.portfolio_returns,
            save_path=f"{prefix}drawdown.png",
            show=show,
        )
        paths.append(self.output_dir / f"{prefix}drawdown.png")

        # Annual returns
        self.annual_returns(
            result.portfolio_returns,
            result.benchmark_returns,
            save_path=f"{prefix}annual_returns.png",
            show=show,
        )
        paths.append(self.output_dir / f"{prefix}annual_returns.png")

        # Rolling metrics
        self.rolling_metrics(
            result.portfolio_returns,
            save_path=f"{prefix}rolling_metrics.png",
            show=show,
        )
        paths.append(self.output_dir / f"{prefix}rolling_metrics.png")

        logger.info(f"Generated {len(paths)} charts")
        return paths
