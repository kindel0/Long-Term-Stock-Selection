"""
Command-line interface for the stock trading system.

Provides commands for:
- Running backtests
- Paper trading
- Live trading
- Tax reporting
"""

import logging
import sys
from datetime import datetime
from pathlib import Path

import click
import pandas as pd

from .config import (
    setup_logging,
    DEFAULT_PANEL_PATH,
    BACKTEST_DEFAULTS,
    PROJECT_ROOT,
)

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
def cli(verbose):
    """Stock Trading System - ML-based stock selection with IBKR integration."""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)


@cli.command()
@click.option(
    "--data",
    "-d",
    type=click.Path(exists=True),
    default=str(DEFAULT_PANEL_PATH),
    help="Path to panel dataset",
)
@click.option(
    "--start",
    type=str,
    default=str(BACKTEST_DEFAULTS["start_year"]),
    help="Start year or date (YYYY or YYYY-MM-DD)",
)
@click.option(
    "--end",
    type=str,
    default=str(BACKTEST_DEFAULTS["end_year"]),
    help="End year or date",
)
@click.option(
    "--capital",
    type=float,
    default=BACKTEST_DEFAULTS["initial_capital"],
    help="Initial capital",
)
@click.option(
    "--stocks",
    "-n",
    type=int,
    default=BACKTEST_DEFAULTS["n_stocks"],
    help="Number of stocks in portfolio",
)
@click.option(
    "--min-cap",
    type=click.Choice(["Nano Cap", "Micro Cap", "Small Cap", "Mid Cap", "Large Cap"]),
    default=BACKTEST_DEFAULTS["min_market_cap"],
    help="Minimum market cap filter",
)
@click.option("--output", "-o", type=click.Path(), help="Output path for results")
@click.option("--plot", is_flag=True, help="Generate charts")
@click.option(
    "--benchmark-source",
    type=click.Choice(["simfin", "yfinance"]),
    default="simfin",
    help="Benchmark data source (simfin=universe, yfinance=S&P500)",
)
@click.option(
    "--benchmark-weighting",
    type=click.Choice(["cap_weighted", "equal_weighted"]),
    default="cap_weighted",
    help="Weighting for SimFin benchmark",
)
def backtest(data, start, end, capital, stocks, min_cap, output, plot, benchmark_source, benchmark_weighting):
    """Run backtest simulation."""
    from .models.stock_selection_rf import StockSelectionRF
    from .backtest.engine import BacktestEngine
    from .reports.charts import ChartGenerator

    click.echo("=" * 60)
    click.echo("RUNNING BACKTEST")
    click.echo("=" * 60)

    # Parse dates
    if len(start) == 4:
        start_date = datetime(int(start), 1, 1)
    else:
        start_date = datetime.strptime(start, "%Y-%m-%d")

    if len(end) == 4:
        end_date = datetime(int(end), 12, 31)
    else:
        end_date = datetime.strptime(end, "%Y-%m-%d")

    click.echo(f"Data: {data}")
    click.echo(f"Period: {start_date.date()} to {end_date.date()}")
    click.echo(f"Capital: ${capital:,.0f}")
    click.echo(f"Stocks: {stocks}")
    click.echo(f"Min Cap: {min_cap}")
    click.echo(f"Benchmark: {benchmark_source} ({benchmark_weighting})")

    # Load data
    click.echo("\nLoading data...")
    df = pd.read_csv(data)
    df["public_date"] = pd.to_datetime(df["public_date"])
    click.echo(f"Loaded {len(df)} rows")

    # Initialize model and engine
    model = StockSelectionRF()
    engine = BacktestEngine(model)

    # Run backtest
    click.echo("\nRunning backtest...")
    result = engine.run(
        data=df,
        start_date=start_date,
        end_date=end_date,
        n_stocks=stocks,
        initial_capital=capital,
        min_market_cap=min_cap,
        target_col="1yr_return",
        benchmark_source=benchmark_source,
        benchmark_weighting=benchmark_weighting,
    )

    # Print results
    engine.print_summary(result)

    # Generate charts
    if plot:
        click.echo("\nGenerating charts...")
        charts = ChartGenerator()
        charts.generate_all(result, show=True)

    # Save results
    if output:
        result_df = result.to_dataframe()
        result_df.to_csv(output, index=False)
        click.echo(f"\nSaved results: {output}")


@cli.command()
@click.option("--data", "-d", type=click.Path(exists=True), help="Path to panel data")
@click.option("--stocks", "-n", type=int, default=15, help="Number of stocks")
@click.option("--dry-run", is_flag=True, help="Show orders without executing")
def paper_trade(data, stocks, dry_run):
    """Run paper trading mode."""
    from .models.stock_selection_rf import StockSelectionRF
    from .trading.order_generator import OrderGenerator
    from .trading.execution_engine import ExecutionEngine, ExecutionMode

    click.echo("=" * 60)
    click.echo("PAPER TRADING MODE")
    click.echo("=" * 60)

    if not data:
        data = str(DEFAULT_PANEL_PATH)

    click.echo(f"Data: {data}")
    click.echo(f"Target stocks: {stocks}")

    # Load data
    df = pd.read_csv(data)
    df["public_date"] = pd.to_datetime(df["public_date"])

    # Get latest date
    latest_date = df["public_date"].max()
    click.echo(f"Latest data: {latest_date.date()}")

    # TODO: Get current positions from IBKR
    # For now, assume empty portfolio
    current_positions = pd.DataFrame()

    # Train model and get selections
    click.echo("\nTraining model...")
    model = StockSelectionRF()
    feature_cols = model.prepare_features(df)

    # Use recent training data
    train_end = latest_date - pd.DateOffset(months=3)
    train_df = df[df["public_date"] <= train_end].copy()

    X_train = train_df[feature_cols]
    y_train = train_df["1yr_return"] if "1yr_return" in train_df.columns else train_df["3mo_return"]
    meta_train = train_df[["sector", "public_date", "TICKER"]]

    model.train(X_train, y_train, meta_train)

    # Get current universe
    test_df = df[df["public_date"] == latest_date].copy()
    X_test = test_df[feature_cols]
    meta_test = test_df[["sector", "public_date", "TICKER", "MthCap"]]

    # Handle missing
    for col in model.feature_columns:
        if col not in X_test.columns:
            X_test[col] = 0
    X_test = X_test[model.feature_columns].fillna(model.feature_medians_).fillna(0)

    # Select stocks
    target_portfolio = model.select_stocks(X_test, meta_test, n=stocks)
    click.echo(f"\nSelected {len(target_portfolio)} stocks:")
    for _, row in target_portfolio.iterrows():
        click.echo(f"  {row['TICKER']}: rank={row['predicted_rank']:.3f}")

    # TODO: Fetch current prices
    # For now, use placeholder prices
    prices = {ticker: 100.0 for ticker in target_portfolio["TICKER"]}

    # Generate orders
    generator = OrderGenerator()
    orders = generator.generate_initial_orders(
        target_portfolio=target_portfolio,
        account_value=30000,
        prices=prices,
    )

    # Execute
    mode = ExecutionMode.DRY_RUN if dry_run else ExecutionMode.PAPER
    engine = ExecutionEngine(mode=mode)
    engine.set_orders(orders)
    engine.review_orders()

    if not dry_run:
        if engine.get_approval():
            engine.execute()
            engine.print_execution_report()
        else:
            click.echo("Execution cancelled")


@cli.command()
@click.option("--data", "-d", type=click.Path(exists=True), help="Path to panel data")
@click.option("--stocks", "-n", type=int, default=15, help="Number of stocks")
@click.option("--confirm", is_flag=True, required=True, help="Confirm live trading")
def live_trade(data, stocks, confirm):
    """Run live trading mode (requires --confirm flag)."""
    if not confirm:
        click.echo("ERROR: Live trading requires --confirm flag")
        sys.exit(1)

    click.echo("=" * 60)
    click.echo("LIVE TRADING MODE")
    click.echo("=" * 60)
    click.echo("\nWARNING: This will execute real trades!")

    click.echo("\nLive trading not yet implemented.")
    click.echo("Use paper-trade mode for testing.")


@cli.command()
@click.option("--year", type=int, default=datetime.now().year, help="Tax year")
@click.option("--format", type=click.Choice(["console", "json", "csv"]), default="console")
@click.option("--output", "-o", type=click.Path(), help="Output path")
def tax_report(year, format, output):
    """Generate tax report for Ireland Revenue."""
    from .tax.tax_reporter import TaxReporter

    click.echo("=" * 60)
    click.echo(f"TAX REPORT - {year}")
    click.echo("=" * 60)

    reporter = TaxReporter()

    if format == "console":
        reporter.print_report(year)
    elif format == "json":
        report = reporter.generate_annual_report(year)
        path = reporter.export_json(report, output)
        click.echo(f"Saved: {path}")
    elif format == "csv":
        report = reporter.generate_annual_report(year)
        paths = reporter.export_csv(report)
        for name, path in paths.items():
            click.echo(f"Saved {name}: {path}")


@cli.command()
@click.option("--rebuild", is_flag=True, help="Rebuild from raw SimFin data")
@click.option("--add-macro", is_flag=True, default=True, help="Add FRED macro features")
@click.option("--output", "-o", type=click.Path(), default=str(DEFAULT_PANEL_PATH))
def build_panel(rebuild, add_macro, output):
    """Build or rebuild the panel dataset."""
    from .data.panel_builder import PanelBuilder

    click.echo("=" * 60)
    click.echo("BUILDING PANEL DATASET")
    click.echo("=" * 60)

    builder = PanelBuilder()
    panel = builder.build(add_macro=add_macro)
    builder.save(panel, output)

    click.echo(f"\nSaved: {output}")
    click.echo(f"Shape: {panel.shape}")


@cli.command()
@click.option("--data", "-d", type=click.Path(exists=True), default=str(DEFAULT_PANEL_PATH))
def validate_data(data):
    """Validate the panel dataset."""
    from .data.data_validator import DataValidator

    click.echo("=" * 60)
    click.echo("VALIDATING DATA")
    click.echo("=" * 60)

    df = pd.read_csv(data)
    df["public_date"] = pd.to_datetime(df["public_date"])

    validator = DataValidator()
    is_valid, report = validator.validate_panel(df)
    validator.print_report(report)

    # Print summary
    summary = validator.get_data_summary(df)
    click.echo("\nDATA SUMMARY:")
    for key, value in summary.items():
        if isinstance(value, dict):
            click.echo(f"  {key}:")
            for k, v in value.items():
                click.echo(f"    {k}: {v}")
        else:
            click.echo(f"  {key}: {value}")


@cli.command()
def status():
    """Show system status and configuration."""
    click.echo("=" * 60)
    click.echo("SYSTEM STATUS")
    click.echo("=" * 60)

    click.echo(f"\nProject Root: {PROJECT_ROOT}")
    click.echo(f"Panel Data: {DEFAULT_PANEL_PATH}")
    click.echo(f"  Exists: {DEFAULT_PANEL_PATH.exists()}")

    if DEFAULT_PANEL_PATH.exists():
        df = pd.read_csv(DEFAULT_PANEL_PATH, nrows=1)
        click.echo(f"  Columns: {len(df.columns)}")

    # Check dependencies
    click.echo("\nDependencies:")
    deps = [
        "pandas",
        "numpy",
        "sklearn",
        "matplotlib",
        "yfinance",
        "click",
    ]
    for dep in deps:
        try:
            __import__(dep.replace("-", "_"))
            click.echo(f"  {dep}: OK")
        except ImportError:
            click.echo(f"  {dep}: MISSING")

    # Optional deps
    click.echo("\nOptional:")
    optional = ["ib_insync", "reportlab", "plotly"]
    for dep in optional:
        try:
            __import__(dep)
            click.echo(f"  {dep}: OK")
        except ImportError:
            click.echo(f"  {dep}: not installed")


def main():
    """Entry point for CLI."""
    cli()


if __name__ == "__main__":
    main()
