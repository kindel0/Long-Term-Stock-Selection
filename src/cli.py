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
    REBALANCE_TO_TARGET,
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
    type=click.Path(),
    default=str(DEFAULT_PANEL_PATH),
    help="Path to panel dataset",
)
@click.option(
    "--rebuild-panel",
    is_flag=True,
    help="Rebuild panel from SimFin source data before backtesting",
)
@click.option(
    "--simfin-dir",
    type=click.Path(exists=True),
    default="data/simfin",
    help="SimFin source directory (for --rebuild-panel)",
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
@click.option(
    "--rebalance-freq",
    type=click.Choice(["A", "Q", "M"]),
    default="A",
    help="Rebalance frequency: A=annual, Q=quarterly, M=monthly",
)
@click.option(
    "--rebalance-month",
    type=int,
    default=3,
    help="Month for annual rebalancing (1-12, default 3=March)",
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
@click.option(
    "--algorithm",
    type=click.Choice(["ridge", "rf"]),
    default="ridge",
    help="ML algorithm: ridge (recommended) or rf (random forest)",
)
@click.option(
    "--roe-weight",
    type=float,
    default=0.5,
    help="Weight for ROE factor (0-1). 0=pure model, 1=pure ROE. Default 0.5",
)
def backtest(data, rebuild_panel, simfin_dir, start, end, capital, stocks, min_cap, rebalance_freq, rebalance_month, output, plot, benchmark_source, benchmark_weighting, algorithm, roe_weight):
    """Run backtest simulation.

    Use --rebuild-panel to regenerate the panel from SimFin source files
    before running the backtest. This ensures you're using the latest data.

    Generates a comprehensive report in results/backtest_TIMESTAMP/ with:
    - config.json: All parameters used
    - summary.json: Performance metrics
    - periods.csv: Period-by-period results
    - selections.csv: Stock selections with individual returns
    - charts/*.png: Visualization charts
    - report.html: HTML summary report
    - backtest.log: Execution log
    """
    from pathlib import Path
    from .models.stock_selection_rf import StockSelectionRF
    from .backtest.engine import BacktestEngine

    click.echo("=" * 60)
    click.echo("RUNNING BACKTEST")
    click.echo("=" * 60)

    # Rebuild panel if requested
    if rebuild_panel:
        from .data.panel_builder import PanelBuilder

        click.echo("\nRebuilding panel from SimFin source data...")
        simfin_path = Path(simfin_dir)

        # Check required files
        required_files = [
            "us-shareprices-daily.csv",
            "us-income-quarterly.csv",
            "us-balance-quarterly.csv",
            "us-cashflow-quarterly.csv",
        ]
        missing = [f for f in required_files if not (simfin_path / f).exists()]
        if missing:
            click.echo(f"ERROR: Missing required SimFin files: {missing}", err=True)
            click.echo(f"Please ensure files exist in: {simfin_path}", err=True)
            return

        builder = PanelBuilder(simfin_dir=simfin_path)
        panel_df = builder.build(add_macro=True, apply_filters=True, winsorize=True)
        builder.save(panel_df, data)
        click.echo(f"Panel rebuilt: {panel_df.shape[0]:,} rows x {panel_df.shape[1]} columns")
        del panel_df  # Free memory before loading

    # Check if data file exists
    data_path = Path(data)
    if not data_path.exists():
        click.echo(f"ERROR: Panel file not found: {data}", err=True)
        click.echo("Run with --rebuild-panel to generate it from SimFin source files", err=True)
        return

    # Parse dates
    if len(start) == 4:
        start_date = datetime(int(start), 1, 1)
    else:
        start_date = datetime.strptime(start, "%Y-%m-%d")

    if len(end) == 4:
        end_date = datetime(int(end), 12, 31)
    else:
        end_date = datetime.strptime(end, "%Y-%m-%d")

    # Format rebalance description
    freq_names = {"A": "Annual", "Q": "Quarterly", "M": "Monthly"}
    rebal_desc = freq_names.get(rebalance_freq, rebalance_freq)
    if rebalance_freq == "A":
        month_names = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun",
                       "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        rebal_desc += f" ({month_names[rebalance_month]} 15)"

    click.echo(f"\nData: {data}")
    click.echo(f"Period: {start_date.date()} to {end_date.date()}")
    click.echo(f"Rebalancing: {rebal_desc}")
    click.echo(f"Capital: ${capital:,.0f}")
    click.echo(f"Stocks: {stocks}")
    click.echo(f"Min Cap: {min_cap}")
    click.echo(f"Benchmark: {benchmark_source} ({benchmark_weighting})")
    click.echo(f"Algorithm: {algorithm} (ROE weight: {roe_weight})")

    # Load data
    click.echo("\nLoading data...")
    df = pd.read_csv(data)
    df["public_date"] = pd.to_datetime(df["public_date"])
    click.echo(f"Loaded {len(df):,} rows")

    # Initialize model and engine
    model = StockSelectionRF(algorithm=algorithm, roe_weight=roe_weight)
    engine = BacktestEngine(model)

    # Run backtest (target_col auto-selected based on rebalance_freq)
    click.echo("\nRunning backtest...")
    result = engine.run(
        data=df,
        start_date=start_date,
        end_date=end_date,
        rebalance_freq=rebalance_freq,
        rebalance_month=rebalance_month,
        n_stocks=stocks,
        initial_capital=capital,
        min_market_cap=min_cap,
        # target_col auto-selected: M→1mo, Q→3mo, S→6mo, A→1yr
        benchmark_source=benchmark_source,
        benchmark_weighting=benchmark_weighting,
    )

    # Generate comprehensive report
    from .reports.backtest_report import BacktestReporter

    # Determine output directory
    if output:
        report_dir = Path(output).parent / Path(output).stem
    else:
        report_dir = None  # Will use timestamped default

    reporter = BacktestReporter(output_dir=report_dir)

    # Build config dict for report
    # Get the target column that was auto-selected
    target_col = REBALANCE_TO_TARGET.get(rebalance_freq, "1yr_return")
    config = {
        "data_path": data,
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "rebalance_freq": rebalance_freq,
        "rebalance_month": rebalance_month,
        "n_stocks": stocks,
        "initial_capital": capital,
        "min_market_cap": min_cap,
        "target_col": target_col,
        "benchmark_source": benchmark_source,
        "benchmark_weighting": benchmark_weighting,
        "algorithm": algorithm,
        "roe_weight": roe_weight,
    }

    # Generate report
    outputs = reporter.generate_report(result, config)

    # Print summary to console
    reporter.print_summary(result)

    # Show charts interactively if requested
    if plot:
        click.echo("\nDisplaying charts...")
        try:
            import matplotlib.pyplot as plt
            for chart_path in outputs.get("charts", []):
                img = plt.imread(chart_path)
                fig, ax = plt.subplots(figsize=(12, 8))
                ax.imshow(img)
                ax.axis("off")
                plt.show()
        except Exception as e:
            click.echo(f"Could not display charts: {e}")

    click.echo(f"\nReport saved to: {reporter.output_dir}")


@cli.command()
@click.option("--data", "-d", type=click.Path(exists=True), help="Path to panel data")
@click.option("--stocks", "-n", type=int, default=15, help="Number of stocks")
@click.option("--capital", type=float, default=None,
              help="Account value override (default: fetch from IBKR or $30K)")
@click.option("--dry-run", is_flag=True, help="Show orders without executing")
@click.option("--algorithm", type=click.Choice(["ridge", "rf"]), default="ridge",
              help="ML algorithm: ridge (recommended) or rf")
@click.option("--roe-weight", type=float, default=0.5,
              help="Weight for ROE factor (0-1)")
def paper_trade(data, stocks, capital, dry_run, algorithm, roe_weight):
    """Run paper trading mode with IBKR connection."""
    from .models.stock_selection_rf import StockSelectionRF
    from .trading.order_generator import OrderGenerator
    from .trading.execution_engine import ExecutionEngine, ExecutionMode
    from .trading.ibkr_client import IBKRClient

    click.echo("=" * 60)
    click.echo("PAPER TRADING MODE")
    click.echo("=" * 60)

    if not data:
        data = str(DEFAULT_PANEL_PATH)

    # Connect to IBKR
    click.echo("\nConnecting to IBKR (paper account)...")
    ibkr = IBKRClient(mode="paper")
    connected = ibkr.connect()

    if not connected:
        click.echo("ERROR: Could not connect to IBKR.")
        click.echo("Make sure TWS or IB Gateway is running with API enabled.")
        click.echo("  - TWS: File > Global Configuration > API > Settings")
        click.echo("  - Enable 'Enable ActiveX and Socket Clients'")
        click.echo("  - UNCHECK 'Read-Only API' to allow trading")
        click.echo("  - Paper trading port should be 7497")
        if dry_run:
            click.echo("\nContinuing in dry-run mode without IBKR...")
            ibkr = None
        else:
            sys.exit(1)

    # Get account info and positions from IBKR
    if ibkr and ibkr.is_connected:
        account_summary = ibkr.get_account_summary()
        current_positions = ibkr.get_positions()

        if capital:
            # Use override value
            account_value = capital
            click.echo(f"\nUsing specified capital: ${account_value:,.2f}")
        elif account_summary:
            click.echo(f"\nAccount Value: ${account_summary.net_liquidation:,.2f}")
            click.echo(f"Cash Available: ${account_summary.total_cash:,.2f}")
            account_value = account_summary.net_liquidation
        else:
            account_value = 30000
            click.echo(f"\nUsing default account value: ${account_value:,.2f}")

        if not current_positions.empty:
            click.echo(f"\nCurrent Positions ({len(current_positions)}):")
            for _, pos in current_positions.iterrows():
                click.echo(f"  {pos['symbol']}: {pos['shares']:.0f} shares @ ${pos['avg_cost']:.2f}")
        else:
            click.echo("\nNo current positions")
            current_positions = pd.DataFrame()
    else:
        account_value = capital or 30000
        current_positions = pd.DataFrame()

    # Load panel data
    click.echo(f"\nLoading data: {data}")
    df = pd.read_csv(data)
    df["public_date"] = pd.to_datetime(df["public_date"])

    latest_date = df["public_date"].max()
    click.echo(f"Latest data: {latest_date.date()}")

    # Train model
    click.echo(f"\nTraining model (algorithm={algorithm}, roe_weight={roe_weight})...")
    model = StockSelectionRF(algorithm=algorithm, roe_weight=roe_weight)
    feature_cols = model.prepare_features(df)

    train_end = latest_date - pd.DateOffset(months=3)
    train_df = df[df["public_date"] <= train_end].copy()

    X_train = train_df[feature_cols]
    return_cols = ["3mo_return", "1yr_return", "6mo_return", "1mo_return"]
    target_col = next((c for c in return_cols if c in train_df.columns), None)
    if target_col is None:
        raise click.ClickException("No return column found in data")
    y_train = train_df[target_col]
    meta_train = train_df[["sector", "public_date", "TICKER"]]

    model.train(X_train, y_train, meta_train)

    # Get current universe and select stocks
    test_df = df[df["public_date"] == latest_date].copy()
    X_test = test_df[feature_cols].copy()
    meta_test = test_df[["sector", "public_date", "TICKER", "MthCap"]].copy()

    # Add ROE for factor weighting
    if "roe" in test_df.columns and "roe" not in X_test.columns:
        X_test["roe"] = test_df.loc[X_test.index, "roe"].values

    for col in model.feature_columns:
        if col not in X_test.columns:
            X_test[col] = 0
    X_test_clean = X_test[model.feature_columns].fillna(model.feature_medians_).fillna(0)

    if "roe" in test_df.columns:
        X_test_clean["roe"] = test_df.loc[X_test_clean.index, "roe"].values

    target_portfolio = model.select_stocks(X_test_clean, meta_test, n=stocks)
    click.echo(f"\nSelected {len(target_portfolio)} stocks:")
    for _, row in target_portfolio.iterrows():
        click.echo(f"  {row['TICKER']}: rank={row['predicted_rank']:.3f}")

    # Fetch current prices from IBKR
    target_symbols = list(target_portfolio["TICKER"])
    current_symbols = list(current_positions["symbol"]) if not current_positions.empty else []
    all_symbols = list(set(target_symbols + current_symbols))

    if ibkr and ibkr.is_connected:
        click.echo(f"\nFetching prices for {len(all_symbols)} symbols...")
        prices = ibkr.get_prices(all_symbols)
        click.echo(f"  Got prices for {len(prices)} symbols")

        # Fill missing with panel prices if available
        for symbol in all_symbols:
            if symbol not in prices:
                panel_price = test_df.loc[test_df["TICKER"] == symbol, "MthPrc"]
                if not panel_price.empty:
                    prices[symbol] = panel_price.iloc[0]
                    click.echo(f"  Using panel price for {symbol}: ${prices[symbol]:.2f}")
                else:
                    prices[symbol] = 100.0
    else:
        # Use panel prices
        prices = {}
        for symbol in all_symbols:
            panel_price = test_df.loc[test_df["TICKER"] == symbol, "MthPrc"]
            if not panel_price.empty:
                prices[symbol] = panel_price.iloc[0]
            else:
                prices[symbol] = 100.0

    # Generate rebalancing orders
    generator = OrderGenerator()
    orders = generator.generate_rebalance_orders(
        current_positions=current_positions,
        target_portfolio=target_portfolio,
        account_value=account_value,
        prices=prices,
    )

    if not orders:
        click.echo("\nNo orders needed - portfolio is already at target.")
        if ibkr and ibkr.is_connected:
            ibkr.disconnect()
        return

    # Execute
    mode = ExecutionMode.DRY_RUN if dry_run else ExecutionMode.PAPER
    engine = ExecutionEngine(mode=mode, ibkr_client=ibkr if not dry_run else None)
    engine.set_orders(orders)
    engine.review_orders()

    if not dry_run:
        if engine.get_approval():
            engine.execute()
            engine.print_execution_report()
        else:
            click.echo("Execution cancelled")

    # Disconnect
    if ibkr and ibkr.is_connected:
        ibkr.disconnect()
        click.echo("\nDisconnected from IBKR")


@cli.command()
@click.option("--data", "-d", type=click.Path(exists=True), help="Path to panel data")
@click.option("--stocks", "-n", type=int, default=15, help="Number of stocks")
@click.option("--confirm", is_flag=True, required=True, help="Confirm live trading")
@click.option("--algorithm", type=click.Choice(["ridge", "rf"]), default="ridge",
              help="ML algorithm: ridge (recommended) or rf")
@click.option("--roe-weight", type=float, default=0.5,
              help="Weight for ROE factor (0-1)")
def live_trade(data, stocks, confirm, algorithm, roe_weight):
    """Run live trading mode (requires --confirm flag)."""
    from .models.stock_selection_rf import StockSelectionRF
    from .trading.order_generator import OrderGenerator
    from .trading.execution_engine import ExecutionEngine, ExecutionMode
    from .trading.ibkr_client import IBKRClient

    if not confirm:
        click.echo("ERROR: Live trading requires --confirm flag")
        sys.exit(1)

    click.echo("=" * 60)
    click.echo("LIVE TRADING MODE")
    click.echo("=" * 60)
    click.echo("\n*** WARNING: This will execute REAL trades! ***\n")

    if not data:
        data = str(DEFAULT_PANEL_PATH)

    # Connect to IBKR LIVE
    click.echo("Connecting to IBKR (LIVE account)...")
    ibkr = IBKRClient(mode="live")
    connected = ibkr.connect()

    if not connected:
        click.echo("ERROR: Could not connect to IBKR live account.")
        click.echo("Make sure TWS or IB Gateway is running with API enabled.")
        click.echo("  - UNCHECK 'Read-Only API' to allow trading")
        click.echo("  - Live trading port should be 7496")
        sys.exit(1)

    # Get account info
    account_summary = ibkr.get_account_summary()
    current_positions = ibkr.get_positions()

    if account_summary:
        click.echo(f"\nAccount Value: ${account_summary.net_liquidation:,.2f}")
        click.echo(f"Cash Available: ${account_summary.total_cash:,.2f}")
        account_value = account_summary.net_liquidation
    else:
        click.echo("ERROR: Could not get account summary")
        ibkr.disconnect()
        sys.exit(1)

    if not current_positions.empty:
        click.echo(f"\nCurrent Positions ({len(current_positions)}):")
        for _, pos in current_positions.iterrows():
            click.echo(f"  {pos['symbol']}: {pos['shares']:.0f} shares @ ${pos['avg_cost']:.2f}")
    else:
        click.echo("\nNo current positions")
        current_positions = pd.DataFrame()

    # Load panel data
    click.echo(f"\nLoading data: {data}")
    df = pd.read_csv(data)
    df["public_date"] = pd.to_datetime(df["public_date"])

    latest_date = df["public_date"].max()
    click.echo(f"Latest data: {latest_date.date()}")

    # Train model
    click.echo(f"\nTraining model (algorithm={algorithm}, roe_weight={roe_weight})...")
    model = StockSelectionRF(algorithm=algorithm, roe_weight=roe_weight)
    feature_cols = model.prepare_features(df)

    train_end = latest_date - pd.DateOffset(months=3)
    train_df = df[df["public_date"] <= train_end].copy()

    X_train = train_df[feature_cols]
    return_cols = ["3mo_return", "1yr_return", "6mo_return", "1mo_return"]
    target_col = next((c for c in return_cols if c in train_df.columns), None)
    if target_col is None:
        raise click.ClickException("No return column found in data")
    y_train = train_df[target_col]
    meta_train = train_df[["sector", "public_date", "TICKER"]]

    model.train(X_train, y_train, meta_train)

    # Get current universe and select stocks
    test_df = df[df["public_date"] == latest_date].copy()
    X_test = test_df[feature_cols].copy()
    meta_test = test_df[["sector", "public_date", "TICKER", "MthCap"]].copy()

    if "roe" in test_df.columns and "roe" not in X_test.columns:
        X_test["roe"] = test_df.loc[X_test.index, "roe"].values

    for col in model.feature_columns:
        if col not in X_test.columns:
            X_test[col] = 0
    X_test_clean = X_test[model.feature_columns].fillna(model.feature_medians_).fillna(0)

    if "roe" in test_df.columns:
        X_test_clean["roe"] = test_df.loc[X_test_clean.index, "roe"].values

    target_portfolio = model.select_stocks(X_test_clean, meta_test, n=stocks)
    click.echo(f"\nSelected {len(target_portfolio)} stocks:")
    for _, row in target_portfolio.iterrows():
        click.echo(f"  {row['TICKER']}: rank={row['predicted_rank']:.3f}")

    # Fetch prices
    target_symbols = list(target_portfolio["TICKER"])
    current_symbols = list(current_positions["symbol"]) if not current_positions.empty else []
    all_symbols = list(set(target_symbols + current_symbols))

    click.echo(f"\nFetching prices for {len(all_symbols)} symbols...")
    prices = ibkr.get_prices(all_symbols)
    click.echo(f"  Got prices for {len(prices)} symbols")

    # Fill missing
    for symbol in all_symbols:
        if symbol not in prices:
            panel_price = test_df.loc[test_df["TICKER"] == symbol, "MthPrc"]
            if not panel_price.empty:
                prices[symbol] = panel_price.iloc[0]
                click.echo(f"  Using panel price for {symbol}: ${prices[symbol]:.2f}")
            else:
                click.echo(f"  WARNING: No price for {symbol}, skipping")

    # Generate rebalancing orders
    generator = OrderGenerator()
    orders = generator.generate_rebalance_orders(
        current_positions=current_positions,
        target_portfolio=target_portfolio,
        account_value=account_value,
        prices=prices,
    )

    if not orders:
        click.echo("\nNo orders needed - portfolio is already at target.")
        ibkr.disconnect()
        return

    # Execute
    engine = ExecutionEngine(mode=ExecutionMode.LIVE, ibkr_client=ibkr)
    engine.set_orders(orders)
    engine.review_orders()

    # Extra confirmation for live trading
    click.echo("\n" + "!" * 60)
    click.echo("!!! FINAL CONFIRMATION - LIVE TRADING !!!")
    click.echo("!" * 60)

    if engine.get_approval():
        engine.execute()
        engine.print_execution_report()
    else:
        click.echo("Execution cancelled")

    # Disconnect
    ibkr.disconnect()
    click.echo("\nDisconnected from IBKR")


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
@click.option(
    "--simfin-dir",
    type=click.Path(exists=True),
    default="data/simfin",
    help="Directory with SimFin CSV files",
)
@click.option("--add-macro/--no-macro", default=True, help="Add FRED macro features")
@click.option("--filters/--no-filters", default=True, help="Apply quality filters")
@click.option("--winsorize/--no-winsorize", default=True, help="Winsorize ratio columns")
@click.option(
    "--format",
    type=click.Choice(["csv", "parquet"]),
    default="csv",
    help="Output format",
)
@click.option("--output", "-o", type=click.Path(), default=str(DEFAULT_PANEL_PATH))
def build_panel(simfin_dir, add_macro, filters, winsorize, format, output):
    """Build panel dataset from SimFin source files.

    Required source files in --simfin-dir:
    - us-shareprices-daily.csv
    - us-income-quarterly.csv
    - us-balance-quarterly.csv
    - us-cashflow-quarterly.csv
    - us-companies.csv
    - industries.csv
    """
    from pathlib import Path
    from .data.panel_builder import PanelBuilder

    click.echo("=" * 60)
    click.echo("BUILDING PANEL DATASET")
    click.echo("=" * 60)

    simfin_path = Path(simfin_dir)
    click.echo(f"SimFin source: {simfin_path}")

    # Check required files
    required_files = [
        "us-shareprices-daily.csv",
        "us-income-quarterly.csv",
        "us-balance-quarterly.csv",
        "us-cashflow-quarterly.csv",
    ]
    missing = [f for f in required_files if not (simfin_path / f).exists()]
    if missing:
        click.echo(f"ERROR: Missing required files: {missing}", err=True)
        return

    click.echo(f"Output: {output}")
    click.echo(f"Options: macro={add_macro}, filters={filters}, winsorize={winsorize}")
    click.echo()

    builder = PanelBuilder(simfin_dir=simfin_path)
    panel = builder.build(
        add_macro=add_macro,
        apply_filters=filters,
        winsorize=winsorize,
    )

    # Save with specified format
    if format == "parquet":
        output_path = output.replace(".csv", ".parquet") if output.endswith(".csv") else output
        builder.save(panel, output_path, format="parquet")
    else:
        builder.save(panel, output)

    click.echo()
    click.echo("=" * 60)
    click.echo("PANEL COMPLETE")
    click.echo("=" * 60)
    click.echo(f"Saved: {output}")
    click.echo(f"Shape: {panel.shape[0]:,} rows x {panel.shape[1]} columns")
    click.echo(f"Date range: {panel['public_date'].min().date()} to {panel['public_date'].max().date()}")
    click.echo(f"Tickers: {panel['TICKER'].nunique():,}")


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
