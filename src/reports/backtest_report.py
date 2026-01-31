"""
Comprehensive backtest report generation.

Generates detailed reports including:
- Configuration/parameters used
- Period-by-period results with metrics
- Stock selections and individual performance
- Charts saved to disk
- Summary statistics
"""

import json
import logging
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class BacktestReporter:
    """
    Generates comprehensive backtest reports.

    Creates a report directory with:
    - config.json: All parameters used
    - summary.json: High-level metrics
    - periods.csv: Period-by-period results
    - selections.csv: Stock selections with returns
    - charts/*.png: Visualization charts
    - report.html: HTML summary (optional)
    """

    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize the reporter.

        Args:
            output_dir: Base directory for reports. If None, uses
                       results/backtest_YYYYMMDD_HHMMSS/
        """
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path("results") / f"backtest_{timestamp}"

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.charts_dir = self.output_dir / "charts"
        self.charts_dir.mkdir(exist_ok=True)

        # Setup file logging
        self._setup_logging()

    def _setup_logging(self):
        """Setup file logging for the backtest."""
        log_file = self.output_dir / "backtest.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )

        # Add to root logger
        logging.getLogger().addHandler(file_handler)
        logger.info(f"Report directory: {self.output_dir}")

    def save_config(self, config: Dict[str, Any]) -> Path:
        """
        Save backtest configuration.

        Args:
            config: Dictionary of all parameters used

        Returns:
            Path to saved config file
        """
        config_path = self.output_dir / "config.json"

        # Convert non-serializable types
        serializable = {}
        for key, value in config.items():
            if isinstance(value, datetime):
                serializable[key] = value.isoformat()
            elif isinstance(value, Path):
                serializable[key] = str(value)
            elif hasattr(value, "__dict__"):
                serializable[key] = str(value)
            else:
                serializable[key] = value

        with open(config_path, "w") as f:
            json.dump(serializable, f, indent=2, default=str)

        logger.info(f"Saved config: {config_path}")
        return config_path

    def save_periods(self, result) -> Path:
        """
        Save period-by-period results with metrics.

        Args:
            result: BacktestResult object

        Returns:
            Path to saved CSV
        """
        periods_data = []

        cumulative_return = 1.0
        peak = 1.0
        max_drawdown = 0.0

        for i, p in enumerate(result.periods):
            cumulative_return *= (1 + p.portfolio_return)
            peak = max(peak, cumulative_return)
            drawdown = (peak - cumulative_return) / peak

            if drawdown > max_drawdown:
                max_drawdown = drawdown

            # Calculate rolling Sharpe (annualized, using all periods up to now)
            returns_so_far = [pp.portfolio_return for pp in result.periods[:i+1]]
            if len(returns_so_far) > 1:
                mean_ret = np.mean(returns_so_far)
                std_ret = np.std(returns_so_far, ddof=1)
                # Assume annual periods for annual rebalancing
                sharpe = mean_ret / std_ret if std_ret > 0 else 0
            else:
                sharpe = 0

            periods_data.append({
                "period": i + 1,
                "date": p.date.strftime("%Y-%m-%d"),
                "portfolio_return": p.portfolio_return,
                "universe_return": p.benchmark_return,
                "sp500_return": p.sp500_return,
                "excess_vs_universe": p.portfolio_return - p.benchmark_return,
                "excess_vs_sp500": p.portfolio_return - p.sp500_return,
                "cumulative_return": cumulative_return - 1,
                "drawdown": drawdown,
                "max_drawdown_to_date": max_drawdown,
                "rolling_sharpe": sharpe,
                "n_stocks": p.n_stocks,
                "turnover": p.turnover,
                "fees": p.fees_paid,
                "stocks": ", ".join(p.selected_stocks[:10]) + ("..." if len(p.selected_stocks) > 10 else ""),
            })

        df = pd.DataFrame(periods_data)
        periods_path = self.output_dir / "periods.csv"
        df.to_csv(periods_path, index=False)

        logger.info(f"Saved periods: {periods_path}")
        return periods_path

    def save_selections(self, result) -> Path:
        """
        Save detailed stock selections with individual performance.

        Args:
            result: BacktestResult object

        Returns:
            Path to saved CSV
        """
        selections_data = []

        for p in result.periods:
            for ticker in p.selected_stocks:
                actual_return = p.actual_returns.get(ticker, np.nan)
                predicted_rank = p.predictions.get(ticker, np.nan)

                selections_data.append({
                    "date": p.date.strftime("%Y-%m-%d"),
                    "ticker": ticker,
                    "predicted_rank": predicted_rank,
                    "actual_return": actual_return,
                    "beat_benchmark": actual_return > p.benchmark_return if not np.isnan(actual_return) else None,
                    "excess_return": actual_return - p.benchmark_return if not np.isnan(actual_return) else np.nan,
                })

        df = pd.DataFrame(selections_data)
        selections_path = self.output_dir / "selections.csv"
        df.to_csv(selections_path, index=False)

        logger.info(f"Saved selections: {selections_path} ({len(df)} stock-periods)")
        return selections_path

    def save_summary(self, result, config: Dict[str, Any]) -> Path:
        """
        Save summary statistics.

        Args:
            result: BacktestResult object
            config: Configuration dictionary

        Returns:
            Path to saved JSON
        """
        m = result.metrics
        b = result.benchmark_metrics

        # Calculate additional stats
        excess_returns = [
            p.portfolio_return - p.benchmark_return for p in result.periods
        ]
        win_periods = sum(1 for e in excess_returns if e > 0)

        # Stock-level stats
        all_stock_returns = []
        for p in result.periods:
            all_stock_returns.extend(p.actual_returns.values())
        all_stock_returns = [r for r in all_stock_returns if not np.isnan(r)]

        summary = {
            "generated_at": datetime.now().isoformat(),
            "config": {
                "start_date": config.get("start_date", ""),
                "end_date": config.get("end_date", ""),
                "rebalance_freq": config.get("rebalance_freq", ""),
                "rebalance_month": config.get("rebalance_month", ""),
                "n_stocks": config.get("n_stocks", ""),
                "min_market_cap": config.get("min_market_cap", ""),
                "benchmark_source": config.get("benchmark_source", ""),
            },
            "periods": {
                "total": result.n_periods,
                "start": result.periods[0].date.strftime("%Y-%m-%d") if result.periods else "",
                "end": result.periods[-1].date.strftime("%Y-%m-%d") if result.periods else "",
            },
            "portfolio": {
                "total_return": float(m.total_return),
                "annualized_return": float(m.annualized_return),
                "volatility": float(m.volatility),
                "sharpe_ratio": float(m.sharpe_ratio),
                "max_drawdown": float(m.max_drawdown),
                "avg_period_return": float(np.mean([p.portfolio_return for p in result.periods])),
            },
            "universe_benchmark": {
                "total_return": float(b.total_return),
                "annualized_return": float(b.annualized_return),
                "volatility": float(b.volatility),
            },
            "sp500": {
                "total_return": float(result.sp500_metrics.total_return) if result.sp500_metrics else None,
                "annualized_return": float(result.sp500_metrics.annualized_return) if result.sp500_metrics else None,
                "volatility": float(result.sp500_metrics.volatility) if result.sp500_metrics else None,
            },
            "vs_universe_benchmark": {
                "alpha": float(m.alpha),
                "beta": float(m.beta),
                "excess_return": float(m.excess_return),
                "information_ratio": float(m.information_ratio),
                "win_rate": win_periods / len(result.periods) if result.periods else 0,
                "avg_excess_return": float(np.mean(excess_returns)) if excess_returns else 0,
            },
            "vs_sp500": self._calc_vs_sp500_stats(result),
            "trading": {
                "total_fees": float(result.total_fees),
                "avg_turnover": float(result.avg_turnover),
                "total_stocks_selected": sum(p.n_stocks for p in result.periods),
            },
            "stock_level": {
                "total_selections": len(all_stock_returns),
                "avg_stock_return": float(np.mean(all_stock_returns)) if all_stock_returns else 0,
                "median_stock_return": float(np.median(all_stock_returns)) if all_stock_returns else 0,
                "std_stock_return": float(np.std(all_stock_returns)) if all_stock_returns else 0,
                "pct_positive": sum(1 for r in all_stock_returns if r > 0) / len(all_stock_returns) if all_stock_returns else 0,
            },
        }

        summary_path = self.output_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)

        logger.info(f"Saved summary: {summary_path}")
        return summary_path

    def _calc_vs_sp500_stats(self, result) -> Dict[str, Any]:
        """Calculate comparison stats vs S&P 500."""
        if result.sp500_metrics is None or len(result.sp500_returns) == 0:
            return {"available": False}

        excess_returns = [
            p.portfolio_return - p.sp500_return for p in result.periods
        ]
        win_periods = sum(1 for e in excess_returns if e > 0)

        port_total = result.cumulative_returns.iloc[-1] - 1
        sp500_total = result.sp500_cumulative.iloc[-1] - 1

        return {
            "available": True,
            "excess_return": float(port_total - sp500_total),
            "win_rate": win_periods / len(result.periods) if result.periods else 0,
            "avg_excess_return": float(np.mean(excess_returns)) if excess_returns else 0,
        }

    def save_charts(self, result) -> List[Path]:
        """
        Generate and save charts.

        Args:
            result: BacktestResult object

        Returns:
            List of paths to saved chart files
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
        except ImportError:
            logger.warning("matplotlib not installed, skipping charts")
            return []

        plt.style.use("seaborn-v0_8-whitegrid")
        saved_charts = []

        # 1. Cumulative Returns
        fig, ax = plt.subplots(figsize=(12, 6))
        dates = [p.date for p in result.periods]
        cum_port = (1 + pd.Series([p.portfolio_return for p in result.periods])).cumprod()
        cum_bench = (1 + pd.Series([p.benchmark_return for p in result.periods])).cumprod()
        cum_sp500 = (1 + pd.Series([p.sp500_return for p in result.periods])).cumprod()

        ax.plot(dates, cum_port.values, label="Portfolio", linewidth=2)
        ax.plot(dates, cum_bench.values, label="Universe (Mid Cap+)", linewidth=2, alpha=0.7)
        ax.plot(dates, cum_sp500.values, label="S&P 500", linewidth=2, linestyle="--", alpha=0.7)
        ax.set_xlabel("Date")
        ax.set_ylabel("Cumulative Return")
        ax.set_title("Cumulative Returns: Portfolio vs Benchmarks")
        ax.legend()
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        fig.autofmt_xdate()

        chart_path = self.charts_dir / "cumulative_returns.png"
        fig.savefig(chart_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved_charts.append(chart_path)

        # 2. Period Returns Bar Chart
        fig, ax = plt.subplots(figsize=(14, 6))
        x = range(len(result.periods))
        width = 0.25

        port_returns = [p.portfolio_return * 100 for p in result.periods]
        bench_returns = [p.benchmark_return * 100 for p in result.periods]
        sp500_returns = [p.sp500_return * 100 for p in result.periods]

        ax.bar([i - width for i in x], port_returns, width, label="Portfolio")
        ax.bar([i for i in x], bench_returns, width, label="Universe", alpha=0.7)
        ax.bar([i + width for i in x], sp500_returns, width, label="S&P 500", alpha=0.7)

        ax.set_xlabel("Period")
        ax.set_ylabel("Return (%)")
        ax.set_title("Period Returns")
        ax.legend()
        ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)

        # Label x-axis with years
        years = [p.date.year for p in result.periods]
        ax.set_xticks(x)
        ax.set_xticklabels(years, rotation=45)

        chart_path = self.charts_dir / "period_returns.png"
        fig.savefig(chart_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved_charts.append(chart_path)

        # 3. Drawdown Chart
        fig, ax = plt.subplots(figsize=(12, 4))

        cumulative = 1.0
        peak = 1.0
        drawdowns = []

        for p in result.periods:
            cumulative *= (1 + p.portfolio_return)
            peak = max(peak, cumulative)
            drawdowns.append((peak - cumulative) / peak * 100)

        ax.fill_between(dates, 0, drawdowns, alpha=0.5, color="red")
        ax.plot(dates, drawdowns, color="darkred", linewidth=1)
        ax.set_xlabel("Date")
        ax.set_ylabel("Drawdown (%)")
        ax.set_title("Portfolio Drawdown")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.invert_yaxis()
        fig.autofmt_xdate()

        chart_path = self.charts_dir / "drawdown.png"
        fig.savefig(chart_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved_charts.append(chart_path)

        # 4. Excess Returns Distribution
        fig, ax = plt.subplots(figsize=(10, 6))

        excess = [(p.portfolio_return - p.benchmark_return) * 100 for p in result.periods]
        ax.hist(excess, bins=15, edgecolor="black", alpha=0.7)
        ax.axvline(x=0, color="red", linestyle="--", linewidth=2, label="Zero")
        ax.axvline(x=np.mean(excess), color="green", linestyle="-", linewidth=2,
                   label=f"Mean: {np.mean(excess):.1f}%")

        ax.set_xlabel("Excess Return (%)")
        ax.set_ylabel("Frequency")
        ax.set_title("Distribution of Excess Returns (Portfolio - Benchmark)")
        ax.legend()

        chart_path = self.charts_dir / "excess_returns_dist.png"
        fig.savefig(chart_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved_charts.append(chart_path)

        # 5. Rolling Metrics
        if len(result.periods) >= 3:
            fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

            # Rolling return (3-period)
            port_ret = pd.Series([p.portfolio_return for p in result.periods], index=dates)
            rolling_ret = port_ret.rolling(3, min_periods=1).mean() * 100

            axes[0].plot(dates, rolling_ret, linewidth=2)
            axes[0].axhline(y=0, color="black", linestyle="-", linewidth=0.5)
            axes[0].set_ylabel("Rolling Avg Return (%)")
            axes[0].set_title("Rolling 3-Period Average Return")

            # Rolling Sharpe
            rolling_sharpe = port_ret.rolling(3, min_periods=2).apply(
                lambda x: x.mean() / x.std() if x.std() > 0 else 0
            )
            axes[1].plot(dates, rolling_sharpe, linewidth=2, color="orange")
            axes[1].axhline(y=0, color="black", linestyle="-", linewidth=0.5)
            axes[1].set_ylabel("Rolling Sharpe")
            axes[1].set_xlabel("Date")
            axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

            fig.autofmt_xdate()
            fig.suptitle("Rolling Metrics (3-Period Window)")

            chart_path = self.charts_dir / "rolling_metrics.png"
            fig.savefig(chart_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            saved_charts.append(chart_path)

        logger.info(f"Saved {len(saved_charts)} charts to {self.charts_dir}")
        return saved_charts

    def generate_html_report(self, result, config: Dict[str, Any]) -> Path:
        """
        Generate HTML summary report.

        Args:
            result: BacktestResult object
            config: Configuration dictionary

        Returns:
            Path to HTML file
        """
        m = result.metrics
        b = result.benchmark_metrics

        # Calculate stats
        excess_returns = [p.portfolio_return - p.benchmark_return for p in result.periods]
        win_rate = sum(1 for e in excess_returns if e > 0) / len(excess_returns) if excess_returns else 0

        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Backtest Report - {datetime.now().strftime("%Y-%m-%d")}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 40px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        h1 {{ color: #333; border-bottom: 2px solid #007bff; padding-bottom: 10px; }}
        h2 {{ color: #555; margin-top: 30px; }}
        .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin: 20px 0; }}
        .metric-card {{ background: #f8f9fa; padding: 20px; border-radius: 8px; border-left: 4px solid #007bff; }}
        .metric-card.positive {{ border-left-color: #28a745; }}
        .metric-card.negative {{ border-left-color: #dc3545; }}
        .metric-value {{ font-size: 28px; font-weight: bold; color: #333; }}
        .metric-label {{ color: #666; font-size: 14px; margin-top: 5px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background: #f8f9fa; font-weight: 600; }}
        tr:hover {{ background: #f8f9fa; }}
        .positive {{ color: #28a745; }}
        .negative {{ color: #dc3545; }}
        .chart-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(500px, 1fr)); gap: 20px; margin: 20px 0; }}
        .chart-grid img {{ width: 100%; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .config-table {{ font-size: 14px; }}
        .config-table td:first-child {{ font-weight: 600; width: 200px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Backtest Report</h1>
        <p>Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>

        <h2>Configuration</h2>
        <table class="config-table">
            <tr><td>Period</td><td>{config.get('start_date', '')} to {config.get('end_date', '')}</td></tr>
            <tr><td>Rebalancing</td><td>{config.get('rebalance_freq', '')} (Month {config.get('rebalance_month', '')})</td></tr>
            <tr><td>Portfolio Size</td><td>{config.get('n_stocks', '')} stocks</td></tr>
            <tr><td>Min Market Cap</td><td>{config.get('min_market_cap', '')}</td></tr>
            <tr><td>Benchmark</td><td>{config.get('benchmark_source', '')} ({config.get('benchmark_weighting', '')})</td></tr>
            <tr><td>Initial Capital</td><td>${config.get('initial_capital', 0):,.0f}</td></tr>
        </table>

        <h2>Performance Summary</h2>
        <div class="metrics-grid">
            <div class="metric-card {'positive' if m.total_return > 0 else 'negative'}">
                <div class="metric-value">{m.total_return * 100:.1f}%</div>
                <div class="metric-label">Total Return</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{m.annualized_return * 100:.1f}%</div>
                <div class="metric-label">Annualized Return</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{m.sharpe_ratio:.2f}</div>
                <div class="metric-label">Sharpe Ratio</div>
            </div>
            <div class="metric-card negative">
                <div class="metric-value">{m.max_drawdown * 100:.1f}%</div>
                <div class="metric-label">Max Drawdown</div>
            </div>
        </div>

        <h2>vs Benchmark</h2>
        <div class="metrics-grid">
            <div class="metric-card {'positive' if m.excess_return > 0 else 'negative'}">
                <div class="metric-value">{m.excess_return * 100:.1f}%</div>
                <div class="metric-label">Excess Return (Annualized)</div>
            </div>
            <div class="metric-card {'positive' if m.alpha > 0 else 'negative'}">
                <div class="metric-value">{m.alpha * 100:.1f}%</div>
                <div class="metric-label">Alpha</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{m.beta:.2f}</div>
                <div class="metric-label">Beta</div>
            </div>
            <div class="metric-card {'positive' if win_rate > 0.5 else 'negative'}">
                <div class="metric-value">{win_rate * 100:.0f}%</div>
                <div class="metric-label">Win Rate</div>
            </div>
        </div>

        <h2>Charts</h2>
        <div class="chart-grid">
            <img src="charts/cumulative_returns.png" alt="Cumulative Returns">
            <img src="charts/period_returns.png" alt="Period Returns">
            <img src="charts/drawdown.png" alt="Drawdown">
            <img src="charts/excess_returns_dist.png" alt="Excess Returns Distribution">
        </div>

        <h2>Period Results</h2>
        <table>
            <tr>
                <th>Date</th>
                <th>Return</th>
                <th>Benchmark</th>
                <th>Excess</th>
                <th>Cumulative</th>
                <th>Drawdown</th>
                <th>Stocks</th>
                <th>Turnover</th>
            </tr>
"""
        cumulative = 1.0
        peak = 1.0

        for p in result.periods:
            cumulative *= (1 + p.portfolio_return)
            peak = max(peak, cumulative)
            drawdown = (peak - cumulative) / peak

            excess = p.portfolio_return - p.benchmark_return
            excess_class = "positive" if excess > 0 else "negative"

            html += f"""            <tr>
                <td>{p.date.strftime("%Y-%m-%d")}</td>
                <td class="{'positive' if p.portfolio_return > 0 else 'negative'}">{p.portfolio_return * 100:.1f}%</td>
                <td>{p.benchmark_return * 100:.1f}%</td>
                <td class="{excess_class}">{excess * 100:.1f}%</td>
                <td>{(cumulative - 1) * 100:.1f}%</td>
                <td class="negative">{drawdown * 100:.1f}%</td>
                <td>{p.n_stocks}</td>
                <td>{p.turnover * 100:.0f}%</td>
            </tr>
"""

        html += """        </table>

        <h2>Trading Summary</h2>
        <table class="config-table">
            <tr><td>Total Periods</td><td>{n_periods}</td></tr>
            <tr><td>Total Fees</td><td>${total_fees:,.2f}</td></tr>
            <tr><td>Average Turnover</td><td>{avg_turnover:.1f}%</td></tr>
        </table>

        <p style="margin-top: 40px; color: #999; font-size: 12px;">
            Generated by Long-Term Stock Selection System
        </p>
    </div>
</body>
</html>
""".format(
            n_periods=result.n_periods,
            total_fees=result.total_fees,
            avg_turnover=result.avg_turnover * 100,
        )

        html_path = self.output_dir / "report.html"
        with open(html_path, "w") as f:
            f.write(html)

        logger.info(f"Saved HTML report: {html_path}")
        return html_path

    def generate_report(self, result, config: Dict[str, Any]) -> Dict[str, Path]:
        """
        Generate complete backtest report.

        Args:
            result: BacktestResult object
            config: Configuration dictionary

        Returns:
            Dictionary of output file paths
        """
        logger.info("="*60)
        logger.info("GENERATING BACKTEST REPORT")
        logger.info("="*60)

        outputs = {
            "config": self.save_config(config),
            "periods": self.save_periods(result),
            "selections": self.save_selections(result),
            "summary": self.save_summary(result, config),
        }

        # Charts
        chart_paths = self.save_charts(result)
        outputs["charts"] = chart_paths

        # HTML report
        outputs["html"] = self.generate_html_report(result, config)

        logger.info("="*60)
        logger.info(f"Report complete: {self.output_dir}")
        logger.info("="*60)

        return outputs

    def print_summary(self, result) -> None:
        """Print summary to console."""
        m = result.metrics
        b = result.benchmark_metrics

        print("\n" + "=" * 70)
        print("BACKTEST SUMMARY")
        print("=" * 70)

        print(f"\nPeriods: {result.n_periods}")
        print(f"Total Fees: ${result.total_fees:.2f}")
        print(f"Avg Turnover: {result.avg_turnover * 100:.1f}%")

        print("\n--- PORTFOLIO ---")
        print(f"Total Return:      {m.total_return * 100:>8.2f}%")
        print(f"Annualized Return: {m.annualized_return * 100:>8.2f}%")
        print(f"Sharpe Ratio:      {m.sharpe_ratio:>8.2f}")
        print(f"Max Drawdown:      {m.max_drawdown * 100:>8.2f}%")

        print("\n--- BENCHMARK ---")
        print(f"Total Return:      {b.total_return * 100:>8.2f}%")
        print(f"Annualized Return: {b.annualized_return * 100:>8.2f}%")

        print("\n--- VS BENCHMARK ---")
        print(f"Alpha:             {m.alpha * 100:>8.2f}%")
        print(f"Beta:              {m.beta:>8.2f}")
        print(f"Excess Return:     {m.excess_return * 100:>8.2f}%")
        print(f"Win Rate:          {m.win_rate * 100:>8.1f}%")

        print("=" * 70)
        print(f"\nReport saved to: {self.output_dir}")
