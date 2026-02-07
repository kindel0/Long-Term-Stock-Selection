"""
Panel dataset builder for stock selection.

Builds a monthly point-in-time (PIT) panel with proper data alignment
to prevent lookahead bias. Based on improved_simfin_panel.py.

Supports multiple data sources (SimFin, EODHD) via dependency injection.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd

from ..config import (
    SIMFIN_DIR,
    EODHD_DIR,
    DATA_DIR,
    MARKET_CAP_BOUNDARIES,
    MARKET_CAP_BASE_YEAR,
    DATA_PIPELINE,
    QUALITY_FILTERS,
    EPS,
    DEFAULT_DATA_SOURCE,
    DEFAULT_MACRO_SOURCE,
)
from .base_loader import DataLoader
from .simfin_loader import SimFinLoader

logger = logging.getLogger(__name__)


class PanelBuilder:
    """
    Builds the stock selection panel dataset.

    The panel is a monthly dataset with:
    - Point-in-time aligned fundamentals (no lookahead)
    - Financial ratios and features
    - Year-adjusted market cap categories
    - Optional macro features

    Supports multiple data sources via dependency injection:
        - SimFin (default)
        - EODHD

    Usage:
        # Default (SimFin)
        builder = PanelBuilder()
        panel = builder.build()

        # With EODHD
        from src.data.eodhd_loader import EODHDLoader
        builder = PanelBuilder(loader=EODHDLoader())
        panel = builder.build()
    """

    def __init__(
        self,
        simfin_dir: Optional[Path] = None,
        fundamental_lag_months: int = None,
        macro_lag_months: int = None,
        loader: Optional[DataLoader] = None,
        data_source: str = DEFAULT_DATA_SOURCE,
        macro_source: str = DEFAULT_MACRO_SOURCE,
    ):
        """
        Initialize the panel builder.

        Args:
            simfin_dir: Directory with SimFin CSV files (for backward compatibility)
            fundamental_lag_months: Lag for fundamental data (default from config)
            macro_lag_months: Lag for macro data (default from config)
            loader: DataLoader instance (overrides simfin_dir and data_source)
            data_source: Data source name ('simfin' or 'eodhd')
            macro_source: Macro data source ('fred' or 'eodhd')
        """
        # Determine which loader to use
        if loader is not None:
            self.loader = loader
        elif data_source == "eodhd":
            from .eodhd_loader import EODHDLoader
            self.loader = EODHDLoader(EODHD_DIR)
        else:
            # Default to SimFin
            self.simfin_dir = Path(simfin_dir) if simfin_dir else SIMFIN_DIR
            self.loader = SimFinLoader(self.simfin_dir)

        self.data_source = self.loader.get_source_name()
        self.macro_source = macro_source

        self.fundamental_lag = (
            fundamental_lag_months
            if fundamental_lag_months is not None
            else DATA_PIPELINE["fundamental_lag_months"]
        )
        self.macro_lag = (
            macro_lag_months
            if macro_lag_months is not None
            else DATA_PIPELINE["macro_shift_months"]
        )

        logger.info(f"PanelBuilder initialized with data_source={self.data_source}, macro_source={self.macro_source}")

    def build_monthly_prices(self) -> pd.DataFrame:
        """
        Build monthly price panel from daily prices.

        Returns:
            Monthly DataFrame with prices, market cap, dividends
        """
        logger.info("Building monthly prices...")
        px = self.loader.load_prices()

        # If prices don't have Shares Outstanding, try to get from fundamentals
        if "Shares Outstanding" not in px.columns or px["Shares Outstanding"].isna().all():
            logger.info("Shares Outstanding not in price data, loading from fundamentals...")
            shares_df = self.loader.get_shares_outstanding()
            if not shares_df.empty:
                # Drop existing empty Shares Outstanding column if present to avoid suffix issues
                if "Shares Outstanding" in px.columns:
                    px = px.drop(columns=["Shares Outstanding"])

                # Merge shares outstanding with prices using merge_asof
                # Note: merge_asof requires both DataFrames to be sorted by the 'on' key (Date)
                # even when using 'by' parameter for grouping
                shares_df = shares_df.rename(columns={"Ticker": "ticker_shares", "Date": "date_shares"})

                # Sort both by the merge key (Date) for merge_asof to work correctly
                px = px.sort_values("Date")
                shares_df = shares_df.sort_values("date_shares")

                # Use merge_asof to get most recent shares outstanding for each price date
                px = pd.merge_asof(
                    px,
                    shares_df[["ticker_shares", "date_shares", "Shares Outstanding"]],
                    left_on="Date",
                    right_on="date_shares",
                    left_by="Ticker",
                    right_by="ticker_shares",
                    direction="backward",
                )
                px = px.drop(columns=["ticker_shares", "date_shares"], errors="ignore")
                logger.info(f"Merged shares outstanding: {px['Shares Outstanding'].notna().sum()} values")
            else:
                logger.warning("No shares outstanding data available - MthCap will be NaN")
                px["Shares Outstanding"] = np.nan

        # Snap to month-end
        px["public_date"] = (
            px["Date"].dt.to_period("M").dt.to_timestamp("M")
            + pd.offsets.MonthEnd(0)
        )

        # Determine which columns to keep
        price_cols = ["Ticker", "SimFinId", "public_date", "Adj. Close"]
        if "Shares Outstanding" in px.columns:
            price_cols.append("Shares Outstanding")

        # Get last trading day of each month per ticker
        last_in_month = (
            px.drop_duplicates(["Ticker", "public_date"], keep="last")
            .loc[:, [c for c in price_cols if c in px.columns]]
            .rename(columns={"Adj. Close": "MthPrc"})
        )

        # Ensure Shares Outstanding column exists
        if "Shares Outstanding" not in last_in_month.columns:
            last_in_month["Shares Outstanding"] = np.nan

        # Trailing 12-month dividends
        div_m = (
            px.groupby(["Ticker", "public_date"], as_index=False)["Dividend"]
            .sum()
            .rename(columns={"Dividend": "Div_m"})
            .sort_values(["Ticker", "public_date"])
        )
        div_m["Div_ttm"] = (
            div_m.groupby("Ticker")["Div_m"]
            .rolling(12, min_periods=1)
            .sum()
            .reset_index(level=0, drop=True)
        )

        # Merge dividends
        m = last_in_month.merge(
            div_m[["Ticker", "public_date", "Div_ttm"]],
            on=["Ticker", "public_date"],
            how="left",
        )
        m = m.sort_values(["Ticker", "public_date"])

        # Forward-fill shares outstanding within ticker
        m["Shares Outstanding"] = m.groupby("Ticker")["Shares Outstanding"].ffill()

        # Calculate market cap and dividend yield
        m["MthCap"] = m["MthPrc"] * m["Shares Outstanding"]
        m["divyield"] = np.where(
            m["MthPrc"] > 0, m["Div_ttm"].fillna(0.0) / m["MthPrc"], np.nan
        )

        # Calculate forward returns for multiple horizons
        return_horizons = {
            "1mo_return": 1,
            "3mo_return": 3,
            "6mo_return": 6,
            "1yr_return": 12,
        }

        for return_col, months in return_horizons.items():
            logger.info(f"Calculating {return_col} ({months} months forward)...")
            future = m[["Ticker", "public_date", "MthPrc"]].copy()
            future["match_date"] = future["public_date"] - pd.DateOffset(months=months)
            future["match_date"] = future["match_date"] + pd.offsets.MonthEnd(0)

            m = m.merge(
                future[["Ticker", "match_date", "MthPrc"]],
                left_on=["Ticker", "public_date"],
                right_on=["Ticker", "match_date"],
                how="left",
                suffixes=("", "_future"),
            )

            m[return_col] = (m["MthPrc_future"] - m["MthPrc"]) / (m["MthPrc"] + EPS)
            m.loc[m["MthPrc"] <= 0, return_col] = np.nan

            # Clean up merge columns for next iteration
            m = m.drop(columns=["MthPrc_future", "match_date"], errors="ignore")
        m = m.rename(columns={"Ticker": "TICKER", "SimFinId": "gvkey"})

        logger.info(
            f"Built {len(m)} monthly price observations for {m['TICKER'].nunique()} tickers"
        )
        return m

    def build_quarterly_fundamentals(self) -> pd.DataFrame:
        """
        Build quarterly fundamentals with TTM calculations.

        Returns:
            DataFrame with fundamentals and TTM values
        """
        logger.info("Building quarterly fundamentals...")
        fund = self.loader.load_fundamentals()

        # Sort by fiscal period for TTM calculations
        fund = fund.sort_values(["Ticker", "Fiscal Year", "Fiscal Period"])

        # Calculate TTM for flow variables
        flow_cols = [
            "Revenue",
            "Cost of Revenue",
            "Gross Profit",
            "Operating Income (Loss)",
            "Interest Expense, Net",
            "Pretax Income (Loss)",
            "Income Tax (Expense) Benefit, Net",
            "Net Income",
            "Net Income (Common)",
            "Depreciation & Amortization",
            "Research & Development",
            "Net Cash from Operating Activities",
            "Change in Fixed Assets & Intangibles",
            "Dividends Paid",
            "Cash from (Repurchase of) Equity",
        ]

        for col in flow_cols:
            if col in fund.columns:
                fund[f"{col}_ttm"] = (
                    fund.groupby("Ticker")[col]
                    .rolling(4, min_periods=4)
                    .sum()
                    .reset_index(level=0, drop=True)
                )

        # Total debt
        if "Short Term Debt" in fund.columns and "Long Term Debt" in fund.columns:
            fund["total_debt"] = fund["Short Term Debt"].fillna(0) + fund[
                "Long Term Debt"
            ].fillna(0)

        # Re-sort by publish date for PIT merge
        fund = fund.sort_values(["Ticker", "Publish Date"])

        logger.info(f"Built {len(fund)} quarterly fundamental observations")
        return fund

    def merge_with_fundamentals(
        self, monthly: pd.DataFrame, fundamentals: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Merge monthly prices with fundamentals using PIT alignment.

        Args:
            monthly: Monthly price DataFrame
            fundamentals: Quarterly fundamentals DataFrame

        Returns:
            Merged DataFrame with proper PIT alignment
        """
        logger.info("Merging prices with fundamentals (PIT alignment)...")

        left = monthly.dropna(subset=["public_date", "TICKER"]).copy()
        right = fundamentals.dropna(subset=["Publish Date", "Ticker"]).copy()

        # Apply lag for PIT compliance
        if self.fundamental_lag > 0:
            right["Publish Date"] = right["Publish Date"] + pd.DateOffset(
                months=self.fundamental_lag
            )
            logger.info(
                f"Applied {self.fundamental_lag}-month lag to fundamentals for PIT"
            )

        left = left.sort_values(["public_date", "TICKER"]).reset_index(drop=True)
        right = right.sort_values(["Publish Date", "Ticker"]).reset_index(drop=True)

        # Merge using asof to get most recent fundamental data
        out = pd.merge_asof(
            left,
            right,
            left_on="public_date",
            right_on="Publish Date",
            left_by="TICKER",
            right_by="Ticker",
            direction="backward",
            allow_exact_matches=(self.fundamental_lag == 0),
        )

        out = out.sort_values(["TICKER", "public_date"]).reset_index(drop=True)

        # Compute ratios
        out = self._compute_ratios(out)

        # Drop duplicate ticker column
        out = out.drop(columns=["Ticker"], errors="ignore")

        logger.info(f"Merged panel: {len(out)} rows")
        return out

    def _compute_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all financial ratios."""
        eps = EPS

        # Create lagged balance sheet items for averages
        balance_items = [
            "Total Assets",
            "Total Equity",
            "Total Current Assets",
            "Total Current Liabilities",
            "Inventories",
            "Accounts & Notes Receivable",
            "Payables & Accruals",
        ]

        for item in balance_items:
            if item in df.columns:
                df[f"{item}_lag1"] = df.groupby("TICKER")[item].shift(1)

        # Valuation ratios
        df["bm"] = df["Total Equity"] / (df["MthCap"] + eps)
        df["ptb"] = (df["MthCap"] + eps) / (df["Total Equity"] + eps)
        df["ps"] = (df["MthCap"] + eps) / (df["Revenue_ttm"] + eps)
        df["pcf"] = (df["MthCap"] + eps) / (
            df["Net Cash from Operating Activities_ttm"] + eps
        )

        # Profitability
        df["npm"] = df["Net Income_ttm"] / (df["Revenue_ttm"] + eps)
        df["gpm"] = df["Gross Profit_ttm"] / (df["Revenue_ttm"] + eps)
        df["roa"] = df["Net Income_ttm"] / (df["Total Assets"] + eps)
        df["roe"] = df["Net Income_ttm"] / (df["Total Equity"] + eps)

        # Average-based returns
        avg_equity = (df["Total Equity"] + df["Total Equity_lag1"]) / 2
        avg_assets = (df["Total Assets"] + df["Total Assets_lag1"]) / 2
        df["aftret_eq"] = df["Net Income_ttm"] / (avg_equity + eps)
        df["aftret_equity"] = df["Net Income_ttm"] / (avg_equity + eps)

        # EBIT and EBITDA
        df["EBIT_ttm"] = df["Operating Income (Loss)_ttm"]
        df["EBITDA_ttm"] = (
            df["Operating Income (Loss)_ttm"] + df["Depreciation & Amortization_ttm"]
        )

        # ROCE
        capital_employed = df["Total Assets"] - df["Total Current Liabilities"]
        df["roce"] = df["EBIT_ttm"] / (capital_employed + eps)

        # Pre-tax return on NOA
        noa = df["Total Assets"] - df["Total Current Liabilities"]
        noa_lag = df["Total Assets_lag1"] - df["Total Current Liabilities_lag1"]
        avg_noa = (noa + noa_lag) / 2
        df["pretret_noa"] = df["Operating Income (Loss)_ttm"] / (avg_noa + eps)

        # Cash and EV
        df["cash"] = df["Cash, Cash Equivalents & Short Term Investments"]
        df["EV"] = df["MthCap"] + df["total_debt"] - df["cash"].fillna(0)
        df["evm"] = df["EV"] / (df["EBITDA_ttm"] + eps)
        df["debt_ebitda"] = df["total_debt"] / (df["EBITDA_ttm"] + eps)

        # P/E variants
        df["pe_inc"] = df["MthCap"] / (df["Net Income_ttm"] + eps)
        if "Net Income (Common)_ttm" in df.columns:
            df["pe_exi"] = df["MthCap"] / (df["Net Income (Common)_ttm"] + eps)
        else:
            df["pe_exi"] = df["pe_inc"]
        df["pe_op_basic"] = df["MthCap"] / (df["Operating Income (Loss)_ttm"] + eps)
        df["pe_op_dil"] = df["pe_op_basic"]

        # Margins
        df["opmbd"] = df["Operating Income (Loss)_ttm"] / (df["Revenue_ttm"] + eps)
        df["opmad"] = df["EBITDA_ttm"] / (df["Revenue_ttm"] + eps)
        df["ptpm"] = df["Pretax Income (Loss)_ttm"] / (df["Revenue_ttm"] + eps)
        df["cfm"] = df["Net Cash from Operating Activities_ttm"] / (
            df["Revenue_ttm"] + eps
        )

        # Payout
        df["dpr"] = (-df["Dividends Paid_ttm"].fillna(0)) / (df["Net Income_ttm"] + eps)

        # Solvency
        df["de_ratio"] = df["total_debt"] / (df["Total Equity"] + eps)
        df["capital_ratio"] = df["Total Equity"] / (df["Total Assets"] + eps)
        df["debt_at"] = df["total_debt"] / (df["Total Assets"] + eps)
        df["debt_assets"] = df["debt_at"]
        df["debt_capital"] = df["total_debt"] / (
            df["total_debt"] + df["Total Equity"] + eps
        )
        df["dltt_be"] = df["Long Term Debt"] / (df["Total Equity"] + eps)

        # Interest coverage
        df["intcov"] = df["EBIT_ttm"] / (df["Interest Expense, Net_ttm"].abs() + eps)
        df["intcov_ratio"] = df["intcov"]
        df["int_debt"] = df["Interest Expense, Net_ttm"].abs() / (df["total_debt"] + eps)
        df["int_totdebt"] = df["int_debt"]

        # Liquidity
        df["curr_ratio"] = df["Total Current Assets"] / (
            df["Total Current Liabilities"] + eps
        )
        df["quick_ratio"] = (df["Total Current Assets"] - df["Inventories"]) / (
            df["Total Current Liabilities"] + eps
        )
        df["cash_ratio"] = df["cash"] / (df["Total Current Liabilities"] + eps)
        df["cash_conversion"] = df["Net Cash from Operating Activities_ttm"] / (
            df["Net Income_ttm"] + eps
        )

        # Financial soundness
        df["cash_lt"] = df["cash"] / (df["Total Assets"] + eps)
        df["invt_act"] = df["Inventories"] / (df["Total Current Assets"] + eps)
        df["rect_act"] = df["Accounts & Notes Receivable"] / (
            df["Total Current Assets"] + eps
        )
        df["short_debt"] = df["Short Term Debt"] / (df["total_debt"] + eps)
        df["curr_debt"] = df["Short Term Debt"] / (df["total_debt"] + eps)
        df["lt_debt"] = df["Long Term Debt"] / (df["total_debt"] + eps)
        df["profit_lct"] = df["Net Income_ttm"] / (
            df["Total Current Liabilities"] + eps
        )
        df["ocf_lct"] = df["Net Cash from Operating Activities_ttm"] / (
            df["Total Current Liabilities"] + eps
        )
        df["cash_debt"] = df["cash"] / (df["total_debt"] + eps)

        # Free cash flow
        capex = (-df["Change in Fixed Assets & Intangibles_ttm"]).clip(lower=0)
        df["capex_ttm"] = capex
        df["fcf_ttm"] = df["Net Cash from Operating Activities_ttm"] - df["capex_ttm"]
        df["fcf_ocf"] = df["fcf_ttm"] / (
            df["Net Cash from Operating Activities_ttm"] + eps
        )

        # Efficiency (using averaged denominators)
        avg_inventory = (df["Inventories"] + df["Inventories_lag1"]) / 2
        avg_receivables = (
            df["Accounts & Notes Receivable"]
            + df["Accounts & Notes Receivable_lag1"]
        ) / 2
        avg_payables = (
            df["Payables & Accruals"] + df["Payables & Accruals_lag1"]
        ) / 2

        df["at_turn"] = df["Revenue_ttm"] / (avg_assets + eps)
        df["inv_turn"] = df["Cost of Revenue_ttm"] / (avg_inventory + eps)
        df["rect_turn"] = df["Revenue_ttm"] / (avg_receivables + eps)
        df["pay_turn"] = df["Cost of Revenue_ttm"] / (avg_payables + eps)

        # Other efficiency
        df["invcap"] = df["Total Equity"] + df["total_debt"] - df["cash"].fillna(0)
        df["sale_invcap"] = df["Revenue_ttm"] / (df["invcap"] + eps)
        df["sale_equity"] = df["Revenue_ttm"] / (df["Total Equity"] + eps)
        nwc = df["Total Current Assets"] - df["Total Current Liabilities"]
        df["sale_nwc"] = df["Revenue_ttm"] / (nwc + eps)

        # Other
        df["accrual"] = (
            df["Net Income_ttm"] - df["Net Cash from Operating Activities_ttm"]
        ) / (df["Total Assets"] + eps)
        df["GProf"] = df["Gross Profit_ttm"] / (df["Total Assets"] + eps)
        df["rd_sale"] = df["Research & Development_ttm"] / (df["Revenue_ttm"] + eps)
        df["lt_ppent"] = df["Property, Plant & Equipment, Net"] / (
            df["Total Assets"] + eps
        )
        df["efftax"] = df["Income Tax (Expense) Benefit, Net_ttm"] / (
            df["Pretax Income (Loss)_ttm"] + eps
        )

        # Drop lag columns
        lag_cols = [c for c in df.columns if c.endswith("_lag1")]
        df = df.drop(columns=lag_cols, errors="ignore")

        return df

    def add_sector_info(self, panel: pd.DataFrame) -> pd.DataFrame:
        """Add sector information and optionally exclude financials."""
        logger.info("Adding sector information...")

        sector_map = self.loader.get_sector_mapping()
        if sector_map.empty:
            logger.warning("No sector mapping available")
            panel["sector"] = "Unknown"
            return panel

        initial_len = len(panel)
        panel = panel.merge(sector_map, on="TICKER", how="left")
        panel["sector"] = panel["sector"].fillna("Unknown")

        if DATA_PIPELINE.get("exclude_financials", True):
            financial_tickers = self.loader.get_financial_tickers()
            panel = panel[~panel["TICKER"].isin(financial_tickers)]
            panel = panel[panel["sector"] != "Financial Services"]
            logger.info(f"Excluded {initial_len - len(panel)} financial rows")

        return panel

    def apply_year_adjusted_caps(self, panel: pd.DataFrame) -> pd.DataFrame:
        """Apply year-adjusted market cap categories."""
        logger.info("Applying year-adjusted market cap categories...")

        panel["year"] = panel["public_date"].dt.year
        median_by_year = panel.groupby("year")["MthCap"].median().to_dict()

        base_year = MARKET_CAP_BASE_YEAR

        if base_year not in median_by_year:
            # Fall back to static buckets
            logger.warning(f"Base year {base_year} not in data, using static buckets")
            panel["cap"] = panel["MthCap"].apply(self._static_cap_bucket)
        else:

            def adjusted_bucket(row):
                mcap = row["MthCap"]
                year = row["year"]
                if pd.isna(mcap) or year not in median_by_year:
                    return np.nan

                scale = median_by_year[year] / median_by_year[base_year]

                for cap_name in ["Nano Cap", "Micro Cap", "Small Cap", "Mid Cap", "Large Cap"]:
                    if mcap < MARKET_CAP_BOUNDARIES[cap_name] * scale:
                        return cap_name
                return "Mega Cap"

            panel["cap"] = panel.apply(adjusted_bucket, axis=1)

        panel = panel.drop(columns=["year"], errors="ignore")
        return panel

    def _static_cap_bucket(self, mcap: float) -> str:
        """Static market cap bucket assignment."""
        if pd.isna(mcap):
            return np.nan
        for cap_name in ["Nano Cap", "Micro Cap", "Small Cap", "Mid Cap", "Large Cap"]:
            if mcap < MARKET_CAP_BOUNDARIES[cap_name]:
                return cap_name
        return "Mega Cap"

    def apply_quality_filters(self, panel: pd.DataFrame) -> pd.DataFrame:
        """Apply data quality filters."""
        logger.info("Applying data quality filters...")
        initial = len(panel)

        # Penny stocks
        panel = panel[panel["MthPrc"] >= QUALITY_FILTERS["min_price"]]

        # Minimum market cap
        panel = panel[panel["MthCap"] >= QUALITY_FILTERS["min_market_cap"]]

        # Positive book equity
        panel = panel[panel["Total Equity"] > QUALITY_FILTERS["min_book_equity"]]

        # Valid returns - apply bounds to all return columns
        low, high = QUALITY_FILTERS["return_bounds"]
        return_cols = ["1mo_return", "3mo_return", "6mo_return", "1yr_return"]
        for col in return_cols:
            if col in panel.columns:
                panel = panel[
                    (panel[col].between(low, high)) | (panel[col].isna())
                ]

        # Positive revenue
        if "Revenue_ttm" in panel.columns:
            panel = panel[(panel["Revenue_ttm"].isna()) | (panel["Revenue_ttm"] > 0)]

        logger.info(f"Quality filters removed {initial - len(panel)} rows")
        return panel

    def winsorize_ratios(
        self, panel: pd.DataFrame, n_mad: float = None
    ) -> pd.DataFrame:
        """Winsorize ratio columns at median +/- n*MAD."""
        n_mad = n_mad or DATA_PIPELINE.get("winsorize_mad", 5.0)

        ratio_keywords = [
            "ratio", "turn", "ptb", "ps", "pcf", "pe_", "bm", "npm", "gpm",
            "roa", "roe", "evm", "dpr", "opm", "ptpm", "cfm", "roce", "efftax",
            "debt_", "int_", "profit_", "ocf_", "cash_", "fcf_", "accrual",
            "sale_", "rd_sale", "lt_ppent",
        ]

        ratio_cols = [
            col
            for col in panel.columns
            if any(kw in col.lower() for kw in ratio_keywords)
        ]

        logger.info(f"Winsorizing {len(ratio_cols)} ratio columns at {n_mad} MAD")

        for col in ratio_cols:
            if col not in panel.columns:
                continue

            median = panel[col].median()
            mad = (panel[col] - median).abs().median()

            if pd.isna(mad) or mad == 0:
                continue

            lower = median - n_mad * mad
            upper = median + n_mad * mad
            panel[col] = panel[col].clip(lower=lower, upper=upper)

        return panel

    def add_macro_features(
        self,
        panel: pd.DataFrame,
        cache_csv: Optional[Path] = None,
        rebuild: bool = False,
        source: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Add macroeconomic features.

        Args:
            panel: Panel DataFrame
            cache_csv: Path to cache file (for FRED)
            rebuild: Force rebuild from source
            source: Override macro source ('fred' or 'eodhd')

        Returns:
            Panel with macro features added
        """
        source = source or self.macro_source

        if source == "eodhd":
            return self._add_eodhd_macro(panel, rebuild)
        else:
            return self._add_fred_macro(panel, cache_csv, rebuild)

    def _add_fred_macro(
        self,
        panel: pd.DataFrame,
        cache_csv: Optional[Path] = None,
        rebuild: bool = False,
    ) -> pd.DataFrame:
        """Add macroeconomic features from FRED."""
        cache_csv = cache_csv or DATA_DIR / "fred_macro_cache.csv"
        macro = None

        # Try to load from cache
        if cache_csv.exists() and not rebuild:
            logger.info(f"Loading macro data from cache: {cache_csv}")
            macro = pd.read_csv(cache_csv)
            if "public_date" in macro.columns:
                macro["public_date"] = pd.to_datetime(macro["public_date"])
            else:
                date_col = [c for c in macro.columns if "date" in c.lower()]
                if date_col:
                    macro["public_date"] = pd.to_datetime(macro[date_col[0]])

        if macro is None:
            logger.info("Fetching macro data from FRED...")
            start = panel["public_date"].min() - pd.DateOffset(years=2)
            end = panel["public_date"].max()

            try:
                def _fetch_fred(series_id, start, end):
                    """Fetch a single FRED series as a DataFrame."""
                    url = (
                        f"https://fred.stlouisfed.org/graph/fredgraph.csv"
                        f"?id={series_id}"
                        f"&cosd={start.strftime('%Y-%m-%d')}"
                        f"&coed={end.strftime('%Y-%m-%d')}"
                    )
                    df = pd.read_csv(url, index_col=0, parse_dates=True)
                    df.index = pd.to_datetime(df.index) + pd.offsets.MonthEnd(0)
                    # FRED uses '.' for missing values
                    df = df.apply(pd.to_numeric, errors="coerce")
                    return df.ffill()

                fed = _fetch_fred("FEDFUNDS", start, end)
                dgs10 = _fetch_fred("DGS10", start, end)
                cpi = _fetch_fred("USACPIALLMINMEI", start, end)
                gdp = _fetch_fred("GDP", start, end)

                # Calculate rates of change
                cpi_series = cpi.iloc[:, 0]
                gdp_series = gdp.iloc[:, 0]

                macro = pd.DataFrame({
                    "FEDFUNDS": fed.iloc[:, 0],
                    "DGS10": dgs10.iloc[:, 0],
                    "USACPIALLMINMEI": cpi_series,
                    "1mo_inf_rate": cpi_series.pct_change(1),
                    "1yr_inf_rate": cpi_series.pct_change(12),
                    "GDP": gdp_series,
                    "1mo_GDP": gdp_series.pct_change(1),
                    "1yr_GDP": gdp_series.pct_change(12),
                }).reset_index()
                macro = macro.rename(columns={"index": "public_date"})

                # Cache for future use
                cache_csv.parent.mkdir(parents=True, exist_ok=True)
                macro.to_csv(cache_csv, index=False)

            except Exception as e:
                logger.error(f"Failed to fetch FRED data: {e}")
                return panel

        # Apply lag
        if self.macro_lag > 0:
            macro_cols = [
                "FEDFUNDS", "DGS10", "USACPIALLMINMEI", "1mo_inf_rate",
                "1yr_inf_rate", "GDP", "1mo_GDP", "1yr_GDP",
            ]
            existing = [c for c in macro_cols if c in macro.columns]
            macro[existing] = macro[existing].shift(self.macro_lag)
            logger.info(f"Applied {self.macro_lag}-month lag to macro features")

        panel = panel.merge(macro, on="public_date", how="left")
        return panel

    def _add_eodhd_macro(
        self,
        panel: pd.DataFrame,
        rebuild: bool = False,
    ) -> pd.DataFrame:
        """Add macroeconomic features from EODHD cache."""
        from .eodhd_loader import EODHDLoader

        # Load EODHD macro if available
        try:
            eodhd_loader = EODHDLoader(EODHD_DIR)
            macro = eodhd_loader.load_macro()
        except FileNotFoundError:
            logger.warning("EODHD macro data not found, falling back to FRED")
            return self._add_fred_macro(panel, rebuild=rebuild)

        if macro.empty:
            logger.warning("EODHD macro data is empty, falling back to FRED")
            return self._add_fred_macro(panel, rebuild=rebuild)

        logger.info(f"Using EODHD macro data: {len(macro)} records")

        # Ensure date column matches
        if "public_date" not in macro.columns:
            date_col = [c for c in macro.columns if "date" in c.lower()]
            if date_col:
                macro = macro.rename(columns={date_col[0]: "public_date"})

        macro["public_date"] = pd.to_datetime(macro["public_date"])

        # Apply lag
        if self.macro_lag > 0:
            macro_cols = [c for c in macro.columns if c != "public_date"]
            macro[macro_cols] = macro[macro_cols].shift(self.macro_lag)
            logger.info(f"Applied {self.macro_lag}-month lag to EODHD macro features")

        panel = panel.merge(macro, on="public_date", how="left")
        return panel

    def build(
        self,
        add_macro: bool = True,
        apply_filters: bool = True,
        year_adjusted_caps: bool = True,
        winsorize: bool = True,
    ) -> pd.DataFrame:
        """
        Build the complete panel dataset.

        Args:
            add_macro: Whether to add FRED macro features
            apply_filters: Whether to apply quality filters
            year_adjusted_caps: Whether to use year-adjusted market caps
            winsorize: Whether to winsorize ratio columns

        Returns:
            Complete panel DataFrame
        """
        logger.info("="*60)
        logger.info("Building stock selection panel")
        logger.info(f"Data source: {self.data_source}")
        logger.info(f"Macro source: {self.macro_source}")
        logger.info("="*60)

        # Build components
        monthly = self.build_monthly_prices()
        fundamentals = self.build_quarterly_fundamentals()

        # Merge
        panel = self.merge_with_fundamentals(monthly, fundamentals)

        # Add sector info
        panel = self.add_sector_info(panel)

        # Clean return columns
        return_cols = ["1mo_return", "3mo_return", "6mo_return", "1yr_return"]
        for col in return_cols:
            if col in panel.columns:
                panel[col] = panel[col].replace([np.inf, -np.inf], np.nan)

        # Quality filters
        if apply_filters:
            panel = self.apply_quality_filters(panel)

        # Market cap categories
        if year_adjusted_caps:
            panel = self.apply_year_adjusted_caps(panel)
        else:
            panel["cap"] = panel["MthCap"].apply(self._static_cap_bucket)

        # Winsorize
        if winsorize:
            panel = self.winsorize_ratios(panel)

        # Macro features
        if add_macro:
            panel = self.add_macro_features(panel)

        # Final cleanup
        panel = panel.dropna(subset=["gvkey", "public_date", "MthPrc", "MthCap"])
        panel = panel.sort_values(["TICKER", "public_date"]).reset_index(drop=True)

        logger.info("="*60)
        logger.info(f"Panel complete: {panel.shape[0]} rows x {panel.shape[1]} columns")
        logger.info(f"Date range: {panel['public_date'].min()} to {panel['public_date'].max()}")
        logger.info(f"Unique tickers: {panel['TICKER'].nunique()}")
        logger.info("="*60)

        return panel

    def save(self, panel: pd.DataFrame, path: str, format: str = "csv") -> None:
        """
        Save panel to disk.

        Args:
            panel: Panel DataFrame
            path: Output path
            format: 'csv' or 'parquet'
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if format == "parquet":
            panel.to_parquet(path, index=False)
        else:
            panel.to_csv(path, index=False)

        logger.info(f"Saved panel to {path}")
