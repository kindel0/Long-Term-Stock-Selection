import os
import argparse
import numpy as np
import pandas as pd
from typing import Optional

# ------------------------------------------------------------
# SimFin -> monthly PIT panel for Long-Term-Stock-Selection
# Enhanced version with improved paper compliance
#
# Key improvements:
# 1. Proper point-in-time (PIT) alignment with configurable lag
# 2. Year-adjusted market cap categories (following Wynne 2023)
# 3. Additional financial ratios using averaged balance sheet items
# 4. Winsorization of extreme outliers
# 5. Data quality filters
# 6. Financial company exclusion
# 7. Stricter TTM requirements
# 8. CORRECTED: Return calculation handles data gaps
# 9. CORRECTED: TTM sorting uses Fiscal periods, not Publish dates
# ------------------------------------------------------------

EPS = 1e-12

def _read_simfin(path: str, usecols=None) -> pd.DataFrame:
    return pd.read_csv(path, sep=";", usecols=usecols, low_memory=False)

def _to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def _available_cols(path: str):
    return list(pd.read_csv(path, sep=";", nrows=0).columns)


def cap_bucket_static(mcap: float):
    """Static market cap buckets (not year-adjusted)"""
    if pd.isna(mcap):
        return np.nan
    if mcap < 50e6:
        return "Nano Cap"
    if mcap < 300e6:
        return "Micro Cap"
    if mcap < 2e9:
        return "Small Cap"
    if mcap < 10e9:
        return "Mid Cap"
    if mcap < 200e9:
        return "Large Cap"
    return "Mega Cap"


def cap_bucket_adjusted(mcap: float, year: int, median_by_year: dict, base_year: int = 2021):
    """
    Year-adjusted market cap buckets following Wynne (2023) methodology.
    Scales boundaries based on median market cap each year.
    """
    if pd.isna(mcap) or year not in median_by_year:
        return np.nan
    
    if base_year not in median_by_year:
        # Fallback to static if base year not available
        return cap_bucket_static(mcap)
    
    # Base boundaries (2021 values from Investopedia/paper)
    base_boundaries = {
        'Nano Cap': 50e6,
        'Micro Cap': 300e6,
        'Small Cap': 2e9,
        'Mid Cap': 10e9,
        'Large Cap': 200e9,
    }
    
    # Scale factor based on median market cap evolution
    scale = median_by_year[year] / median_by_year[base_year]
    
    # Apply scaled boundaries
    if mcap < base_boundaries['Nano Cap'] * scale:
        return "Nano Cap"
    if mcap < base_boundaries['Micro Cap'] * scale:
        return "Micro Cap"
    if mcap < base_boundaries['Small Cap'] * scale:
        return "Small Cap"
    if mcap < base_boundaries['Mid Cap'] * scale:
        return "Mid Cap"
    if mcap < base_boundaries['Large Cap'] * scale:
        return "Large Cap"
    return "Mega Cap"


def build_monthly_prices(cache_dir: str) -> pd.DataFrame:
    fp = os.path.join(cache_dir, "us-shareprices-daily.csv")
    usecols = ["Ticker","SimFinId","Date","Adj. Close","Dividend","Shares Outstanding"]
    px = _read_simfin(fp, usecols=usecols)

    px["Date"] = pd.to_datetime(px["Date"], errors="coerce")
    px = px.dropna(subset=["Ticker","SimFinId","Date"])

    px["Adj. Close"] = _to_num(px["Adj. Close"])
    px["Dividend"] = _to_num(px["Dividend"]).fillna(0.0)
    px["Shares Outstanding"] = _to_num(px["Shares Outstanding"])

    px = px.sort_values(["Ticker","Date"])
    
    # FIXED: Snap to month-end immediately to prevent date alignment errors
    px["public_date"] = px["Date"].dt.to_period("M").dt.to_timestamp("M") + pd.offsets.MonthEnd(0)

    # Month-end price & shares (last trading day in month)
    last_in_month = (
        px.drop_duplicates(["Ticker","public_date"], keep="last")
          .loc[:, ["Ticker","SimFinId","public_date","Adj. Close","Shares Outstanding"]]
          .rename(columns={"Adj. Close":"MthPrc"})
    )

    # Trailing-12m dividends for dividend yield
    div_m = (
        px.groupby(["Ticker","public_date"], as_index=False)["Dividend"].sum()
          .rename(columns={"Dividend":"Div_m"})
          .sort_values(["Ticker","public_date"])
    )
    div_m["Div_ttm"] = (
        div_m.groupby("Ticker")["Div_m"]
             .rolling(12, min_periods=1).sum()
             .reset_index(level=0, drop=True)
    )

    m = last_in_month.merge(div_m[["Ticker","public_date","Div_ttm"]], 
                            on=["Ticker","public_date"], how="left")
    m = m.sort_values(["Ticker","public_date"])

    # Forward-fill shares inside ticker
    m["Shares Outstanding"] = m.groupby("Ticker")["Shares Outstanding"].ffill()

    m["MthCap"] = m["MthPrc"] * m["Shares Outstanding"]
    m["divyield"] = np.where(m["MthPrc"] > 0, 
                             m["Div_ttm"].fillna(0.0) / m["MthPrc"], 
                             np.nan)

    # FIXED: Calculate 1-year forward return by matching dates accurately
    # This prevents errors where .shift(-12) blindly grabs the wrong date if rows are missing
    print("Calculating 1-year returns via date matching...")
    future_prices = m[['Ticker', 'public_date', 'MthPrc']].copy()
    
    # We want the price from (Current Date + 1 Year). 
    # So we set a 'match_date' in the temp dataframe to (Future Date - 1 Year)
    # When we merge Current Date == Match Date, we get the future price.
    future_prices['match_date'] = future_prices['public_date'] - pd.DateOffset(years=1)
    future_prices['match_date'] = future_prices['match_date'] + pd.offsets.MonthEnd(0)

    m = m.merge(future_prices[['Ticker', 'match_date', 'MthPrc']], 
                left_on=['Ticker', 'public_date'], 
                right_on=['Ticker', 'match_date'],
                how='left', 
                suffixes=('', '_future'))

    m["1yr_return"] = (m["MthPrc_future"] - m["MthPrc"]) / (m["MthPrc"] + EPS)
    m.loc[m["MthPrc"] <= 0, "1yr_return"] = np.nan
    
    # Cleanup temporary columns
    m.drop(columns=['MthPrc_future', 'match_date'], inplace=True, errors='ignore')

    # Rename to repo-friendly identifiers
    m = m.rename(columns={"Ticker":"TICKER", "SimFinId":"gvkey"})
    return m


def build_quarterly_fundamentals(cache_dir: str) -> pd.DataFrame:
    inc_fp = os.path.join(cache_dir, "us-income-quarterly.csv")
    bal_fp = os.path.join(cache_dir, "us-balance-quarterly.csv")
    cf_fp  = os.path.join(cache_dir, "us-cashflow-quarterly.csv")

    inc_cols = [
        "Ticker","Fiscal Year","Fiscal Period","Publish Date",
        "Revenue","Cost of Revenue","Gross Profit",
        "Operating Income (Loss)",
        "Interest Expense, Net",
        "Pretax Income (Loss)",
        "Income Tax (Expense) Benefit, Net",
        "Net Income","Net Income (Common)",
        "Depreciation & Amortization",
        "Research & Development",
    ]
    bal_cols = [
        "Ticker","Fiscal Year","Fiscal Period","Publish Date",
        "Total Assets","Total Equity",
        "Total Current Assets","Total Current Liabilities",
        "Inventories",
        "Cash, Cash Equivalents & Short Term Investments",
        "Accounts & Notes Receivable",
        "Payables & Accruals",
        "Property, Plant & Equipment, Net",
        "Short Term Debt","Long Term Debt",
        "Total Liabilities",
    ]
    cf_cols  = [
        "Ticker","Fiscal Year","Fiscal Period","Publish Date",
        "Net Cash from Operating Activities",
        "Change in Fixed Assets & Intangibles",
        "Dividends Paid",
        "Cash from (Repurchase of) Equity",
    ]

    inc_av = set(_available_cols(inc_fp))
    bal_av = set(_available_cols(bal_fp))
    cf_av  = set(_available_cols(cf_fp))

    inc = _read_simfin(inc_fp, usecols=[c for c in inc_cols if c in inc_av])
    bal = _read_simfin(bal_fp, usecols=[c for c in bal_cols if c in bal_av])
    cf  = _read_simfin(cf_fp,  usecols=[c for c in cf_cols  if c in cf_av])

    for df in (inc, bal, cf):
        df["Publish Date"] = pd.to_datetime(df["Publish Date"], errors="coerce")
        df.dropna(subset=["Ticker","Publish Date"], inplace=True)

    # Numeric conversion
    for df in (inc, bal, cf):
        for c in df.columns:
            if c not in ("Ticker","Fiscal Year","Fiscal Period","Publish Date"):
                df[c] = _to_num(df[c])

    # Merge by fiscal keys
    f = (
        inc.merge(bal, on=["Ticker","Fiscal Year","Fiscal Period","Publish Date"], how="outer")
           .merge(cf,  on=["Ticker","Fiscal Year","Fiscal Period","Publish Date"], how="outer")
    )
    
    # FIXED: Sort by Fiscal Year/Period for TTM calculations to ensure accounting accuracy
    # (Previous version sorted by Publish Date, which breaks if reports are late)
    f = f.sort_values(["Ticker", "Fiscal Year", "Fiscal Period"])

    # Trailing-4Q sums for flow variables
    flow_cols = [
        "Revenue","Cost of Revenue","Gross Profit",
        "Operating Income (Loss)","Interest Expense, Net",
        "Pretax Income (Loss)","Income Tax (Expense) Benefit, Net",
        "Net Income","Net Income (Common)",
        "Depreciation & Amortization",
        "Research & Development",
        "Net Cash from Operating Activities",
        "Change in Fixed Assets & Intangibles",
        "Dividends Paid",
        "Cash from (Repurchase of) Equity",
    ]
    for col in flow_cols:
        if col in f.columns:
            f[col + "_ttm"] = (
                f.groupby("Ticker")[col]
                 .rolling(4, min_periods=4)  # Require full 4 quarters
                 .sum()
                 .reset_index(level=0, drop=True)
            )

    # Debt helper
    if "Short Term Debt" in f.columns and "Long Term Debt" in f.columns:
        f["total_debt"] = f["Short Term Debt"].fillna(0) + f["Long Term Debt"].fillna(0)

    # FIXED: Re-sort by Publish Date for the upcoming PIT merge
    f = f.sort_values(["Ticker", "Publish Date"])

    return f


def exclude_financial_companies(panel: pd.DataFrame, cache_dir: str) -> pd.DataFrame:
    """
    Update: Merges sector information and excludes financial companies.
    """
    companies_fp = os.path.join(cache_dir, "us-companies.csv")
    industries_fp = os.path.join(cache_dir, "industries.csv")
    
    if not os.path.exists(companies_fp) or not os.path.exists(industries_fp):
        print("Warning: mapping files not found, cannot add sector info or exclude financials")
        return panel
    
    try:
        companies = pd.read_csv(companies_fp, sep=";")
        industries = pd.read_csv(industries_fp, sep=";")
        
        # 1. Map Tickers to Sectors
        comp_ind = companies.merge(industries, on='IndustryId', how='left')
        sector_map = comp_ind[['Ticker', 'Sector']].rename(columns={'Ticker': 'TICKER', 'Sector': 'sector'})
        
        initial_len = len(panel)
        panel = panel.merge(sector_map, on='TICKER', how='left')
        panel['sector'] = panel['sector'].fillna('Unknown')
        
        # 2. Exclude Financials (IndustryId starting with 103)
        financial_tickers = set(companies[companies['IndustryId'].astype(str).str.startswith('103')]['Ticker'])

        # Updated Logic:
        # 1. Remove by Industry ID 103
        panel = panel[~panel['TICKER'].isin(financial_tickers)]

        # 2. Remove anything that still has the "Financial Services" sector label
        panel = panel[panel['sector'] != 'Financial Services']
        
        print(f"Added sector info and excluded {initial_len - len(panel)} financial rows.")
    except Exception as e:
        print(f"Warning: Could not process sector/financial info: {e}")
    
    return panel


def merge_monthly_with_fundamentals(
    monthly: pd.DataFrame, 
    fundamentals: pd.DataFrame,
    lag_months: int = 1
) -> pd.DataFrame:
    """
    Merge monthly prices with quarterly fundamentals using point-in-time alignment.
    """
    left = monthly.dropna(subset=["public_date","TICKER"]).copy()
    right = fundamentals.dropna(subset=["Publish Date","Ticker"]).copy()

    # Apply lag to fundamental publish dates for PIT compliance
    if lag_months > 0:
        right["Publish Date"] = right["Publish Date"] + pd.DateOffset(months=lag_months)
        print(f"Applied {lag_months}-month lag to fundamental publish dates for PIT alignment")

    left = left.sort_values(["public_date","TICKER"]).reset_index(drop=True)
    right = right.sort_values(["Publish Date","Ticker"]).reset_index(drop=True)

    out = pd.merge_asof(
        left,
        right,
        left_on="public_date",
        right_on="Publish Date",
        left_by="TICKER",
        right_by="Ticker",
        direction="backward",
        allow_exact_matches=(lag_months == 0),  # Only allow exact if no lag
    )
    out = out.sort_values(["TICKER","public_date"]).reset_index(drop=True)

    eps = EPS

    # Create lagged balance sheet items for averaged calculations (paper methodology)
    balance_sheet_items = [
        "Total Assets", "Total Equity", "Total Current Assets", 
        "Total Current Liabilities", "Inventories",
        "Accounts & Notes Receivable", "Payables & Accruals"
    ]
    
    for item in balance_sheet_items:
        if item in out.columns:
            out[f"{item}_lag1"] = out.groupby("TICKER")[item].shift(1)

    # ----- Core ratios (v2) -----
    out["bm"] = out["Total Equity"] / (out["MthCap"] + eps)
    out["ptb"] = (out["MthCap"] + eps) / (out["Total Equity"] + eps)
    out["ps"]  = (out["MthCap"] + eps) / (out["Revenue_ttm"] + eps)
    out["pcf"] = (out["MthCap"] + eps) / (out["Net Cash from Operating Activities_ttm"] + eps)

    out["npm"] = out["Net Income_ttm"] / (out["Revenue_ttm"] + eps)
    out["gpm"] = out["Gross Profit_ttm"] / (out["Revenue_ttm"] + eps)

    out["roa"] = out["Net Income_ttm"] / (out["Total Assets"] + eps)
    out["roe"] = out["Net Income_ttm"] / (out["Total Equity"] + eps)

    out["de_ratio"] = out["total_debt"] / (out["Total Equity"] + eps)

    out["curr_ratio"]  = out["Total Current Assets"] / (out["Total Current Liabilities"] + eps)
    out["quick_ratio"] = (out["Total Current Assets"] - out["Inventories"]) / (out["Total Current Liabilities"] + eps)
    out["cash_ratio"]  = out["Cash, Cash Equivalents & Short Term Investments"] / (out["Total Current Liabilities"] + eps)

    # ----- Extended ratios using AVERAGED balance sheet items (paper methodology) -----
    
    # Average assets for turnover ratios
    avg_assets = (out["Total Assets"] + out["Total Assets_lag1"]) / 2
    avg_equity = (out["Total Equity"] + out["Total Equity_lag1"]) / 2
    avg_inventory = (out["Inventories"] + out["Inventories_lag1"]) / 2
    avg_receivables = (out["Accounts & Notes Receivable"] + out["Accounts & Notes Receivable_lag1"]) / 2
    avg_payables = (out["Payables & Accruals"] + out["Payables & Accruals_lag1"]) / 2
    
    # After-tax return on average common equity (paper uses this)
    out["aftret_eq"] = out["Net Income_ttm"] / (avg_equity + eps)
    
    # After-tax return on average total equity
    out["aftret_equity"] = out["Net Income_ttm"] / (avg_equity + eps)
    
    # Pre-tax return on net operating assets
    noa = out['Total Assets'] - out['Total Current Liabilities']
    noa_lag = out['Total Assets_lag1'] - out['Total Current Liabilities_lag1']
    avg_noa = (noa + noa_lag) / 2
    out["pretret_noa"] = out["Operating Income (Loss)_ttm"] / (avg_noa + eps)

    # ----- Cash and EBITDA -----
    out["cash"] = out["Cash, Cash Equivalents & Short Term Investments"]
    out["EBIT_ttm"] = out["Operating Income (Loss)_ttm"]
    out["EBITDA_ttm"] = out["Operating Income (Loss)_ttm"] + out["Depreciation & Amortization_ttm"]

    # EV and EV/EBITDA
    out["EV"] = out["MthCap"] + out["total_debt"] - out["cash"].fillna(0)
    out["evm"] = out["EV"] / (out["EBITDA_ttm"] + eps)
    out["debt_ebitda"] = out["total_debt"] / (out["EBITDA_ttm"] + eps)

    # ----- P/E variants -----
    out["pe_inc"] = out["MthCap"] / (out["Net Income_ttm"] + eps)
    if "Net Income (Common)_ttm" in out.columns:
        out["pe_exi"] = out["MthCap"] / (out["Net Income (Common)_ttm"] + eps)
    else:
        out["pe_exi"] = out["pe_inc"]

    out["pe_op_basic"] = out["MthCap"] / (out["Operating Income (Loss)_ttm"] + eps)
    out["pe_op_dil"] = out["pe_op_basic"]

    # ----- Payout and margins -----
    out["dpr"] = (-out["Dividends Paid_ttm"].fillna(0)) / (out["Net Income_ttm"] + eps)

    out["opmbd"] = out["Operating Income (Loss)_ttm"] / (out["Revenue_ttm"] + eps)
    out["opmad"] = out["EBITDA_ttm"] / (out["Revenue_ttm"] + eps)
    out["ptpm"]  = out["Pretax Income (Loss)_ttm"] / (out["Revenue_ttm"] + eps)
    out["cfm"]   = out["Net Cash from Operating Activities_ttm"] / (out["Revenue_ttm"] + eps)

    # ----- Return on capital employed -----
    capital_employed = out["Total Assets"] - out["Total Current Liabilities"]
    out["roce"] = out["EBIT_ttm"] / (capital_employed + eps)

    # ----- Tax rate -----
    out["efftax"] = out["Income Tax (Expense) Benefit, Net_ttm"] / (out["Pretax Income (Loss)_ttm"] + eps)

    # ----- Capital structure -----
    out["capital_ratio"] = out["Total Equity"] / (out["Total Assets"] + eps)
    out["cash_lt"] = out["cash"] / (out["Total Assets"] + eps)
    out["invt_act"] = out["Inventories"] / (out["Total Current Assets"] + eps)
    out["rect_act"] = out["Accounts & Notes Receivable"] / (out["Total Current Assets"] + eps)
    out["debt_at"] = out["total_debt"] / (out["Total Assets"] + eps)

    out["short_debt"] = out["Short Term Debt"] / (out["total_debt"] + eps)
    out["curr_debt"]  = out["Short Term Debt"] / (out["total_debt"] + eps)
    out["lt_debt"]    = out["Long Term Debt"] / (out["total_debt"] + eps)

    out["dltt_be"] = out["Long Term Debt"] / (out["Total Equity"] + eps)
    out["debt_assets"]  = out["total_debt"] / (out["Total Assets"] + eps)
    out["debt_capital"] = out["total_debt"] / (out["total_debt"] + out["Total Equity"] + eps)

    # ----- Interest coverage -----
    out["intcov"] = out["EBIT_ttm"] / (out["Interest Expense, Net_ttm"].abs() + eps)
    out["intcov_ratio"] = out["intcov"]
    out["int_debt"] = out["Interest Expense, Net_ttm"].abs() / (out["total_debt"] + eps)
    out["int_totdebt"] = out["int_debt"]

    # ----- Profitability vs current liabilities -----
    out["profit_lct"] = out["Net Income_ttm"] / (out["Total Current Liabilities"] + eps)
    out["ocf_lct"]    = out["Net Cash from Operating Activities_ttm"] / (out["Total Current Liabilities"] + eps)

    # ----- Cash ratios -----
    out["cash_debt"] = out["cash"] / (out["total_debt"] + eps)
    out["cash_conversion"] = out["Net Cash from Operating Activities_ttm"] / (out["Net Income_ttm"] + eps)

    # ----- Free cash flow -----
    capex = (-out["Change in Fixed Assets & Intangibles_ttm"]).clip(lower=0)
    out["capex_ttm"] = capex
    out["fcf_ttm"] = out["Net Cash from Operating Activities_ttm"] - out["capex_ttm"]
    out["fcf_ocf"] = out["fcf_ttm"] / (out["Net Cash from Operating Activities_ttm"] + eps)

    # ----- Accruals -----
    out["accrual"] = (out["Net Income_ttm"] - out["Net Cash from Operating Activities_ttm"]) / (out["Total Assets"] + eps)

    # ----- Gross profitability -----
    out["GProf"] = out["Gross Profit_ttm"] / (out["Total Assets"] + eps)

    # ----- Turnover ratios (using AVERAGED balance sheet items) -----
    out["inv_turn"]  = out["Cost of Revenue_ttm"] / (avg_inventory + eps)
    out["rect_turn"] = out["Revenue_ttm"] / (avg_receivables + eps)
    out["pay_turn"]  = out["Cost of Revenue_ttm"] / (avg_payables + eps)
    out["at_turn"]   = out["Revenue_ttm"] / (avg_assets + eps)

    # ----- Sales efficiency -----
    out["invcap"] = out["Total Equity"] + out["total_debt"] - out["cash"].fillna(0)
    out["sale_invcap"] = out["Revenue_ttm"] / (out["invcap"] + eps)
    out["sale_equity"] = out["Revenue_ttm"] / (out["Total Equity"] + eps)
    out["sale_nwc"] = out["Revenue_ttm"] / ((out["Total Current Assets"] - out["Total Current Liabilities"]) + eps)

    # ----- R&D and PPE intensity -----
    out["rd_sale"] = out["Research & Development_ttm"] / (out["Revenue_ttm"] + eps)
    out["lt_ppent"] = out["Property, Plant & Equipment, Net"] / (out["Total Assets"] + eps)

    # ----- Market cap bucket (will be updated with year-adjusted version) -----
    out["cap"] = out["MthCap"].apply(cap_bucket_static)

    # Drop lagged columns and ticker duplicate
    lag_cols = [c for c in out.columns if c.endswith('_lag1')]
    out = out.drop(columns=lag_cols + ["Ticker"], errors="ignore")
    
    return out


def apply_year_adjusted_market_caps(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Apply year-adjusted market cap categories following Wynne (2023).
    """
    panel['year'] = panel['public_date'].dt.year
    
    # Calculate median market cap by year
    median_by_year = panel.groupby('year')['MthCap'].median().to_dict()
    
    print("Applying year-adjusted market cap categories...")
    print(f"Median market cap by year (sample): {dict(list(median_by_year.items())[:5])}")
    
    # Apply adjusted buckets
    panel['cap'] = panel.apply(
        lambda row: cap_bucket_adjusted(row['MthCap'], row['year'], median_by_year),
        axis=1
    )
    
    panel = panel.drop(columns=['year'], errors='ignore')
    
    return panel


def winsorize_ratios(df: pd.DataFrame, ratio_cols: list, n_mad: float = 5) -> pd.DataFrame:
    """
    Winsorize ratios at median ± n*MAD to handle extreme outliers.
    Paper uses 5 MAD (median absolute deviations).
    """
    print(f"Winsorizing {len(ratio_cols)} ratio columns at {n_mad} MAD...")
    
    for col in ratio_cols:
        if col not in df.columns:
            continue
        
        # Calculate robust statistics
        median = df[col].median()
        mad = (df[col] - median).abs().median()
        
        if pd.isna(mad) or mad == 0:
            continue
            
        lower = median - n_mad * mad
        upper = median + n_mad * mad
        
        # Count how many values are winsorized
        n_winsorized = ((df[col] < lower) | (df[col] > upper)).sum()
        
        if n_winsorized > 0:
            df[col] = df[col].clip(lower=lower, upper=upper)
    
    return df


def apply_data_quality_filters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply standard data quality filters following asset pricing best practices.
    """
    initial = len(df)
    print("\nApplying data quality filters...")
    
    # Remove penny stocks (price < $1)
    before = len(df)
    df = df[df['MthPrc'] >= 1.0]
    print(f"  Removed {before - len(df)} penny stock observations (price < $1)")
    
    # Remove stocks with negative or zero book equity
    before = len(df)
    df = df[df['Total Equity'] > 0]
    print(f"  Removed {before - len(df)} observations with non-positive book equity")
    
    # Remove stocks with very small market cap (< $5M, too illiquid)
    before = len(df)
    df = df[df['MthCap'] >= 5e6]
    print(f"  Removed {before - len(df)} observations with market cap < $5M")
    
    # Remove extreme returns (likely data errors or corporate actions)
    before = len(df)
    df = df[(df['1yr_return'].between(-0.99, 10.0)) | (df['1yr_return'].isna())]
    print(f"  Removed {before - len(df)} observations with extreme returns (<-99% or >1000%)")
    
    # Remove observations with negative revenue
    if 'Revenue_ttm' in df.columns:
        before = len(df)
        df = df[(df['Revenue_ttm'].isna()) | (df['Revenue_ttm'] > 0)]
        print(f"  Removed {before - len(df)} observations with negative revenue")
    
    total_removed = initial - len(df)
    print(f"Total filtered: {total_removed} rows ({100*total_removed/initial:.1f}%)")
    
    return df


def add_fred_macro(
    panel: pd.DataFrame,
    cache_csv: str | None = None,
    rebuild_cache: bool = False,
    shift_months: int = 1,  # Changed default to 1
    cpi_series_prefer: str = "USACPIALLMINMEI",
) -> pd.DataFrame:
    """
    Add macro series from FRED and merge on month-end `public_date`.
    Default shift_months=1 to reduce lookahead risk (paper recommendation).
    """
    try:
        from pandas_datareader import data as pdr
    except Exception as e:
        raise RuntimeError(
            "pandas_datareader is required for --add-macro (pip install pandas_datareader)"
        ) from e

    def _to_month_end_index(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        # 1. Ensure the index is a DatetimeIndex
        out.index = pd.to_datetime(out.index)
        
        # 2. Snap every date to the end of its current month
        # Using + MonthEnd(0) moves mid-month dates to the end, 
        # and leaves existing month-end dates as they are.
        out.index = out.index + pd.offsets.MonthEnd(0)
        
        # 3. Standardize sort and deduplicate
        out = out.sort_index()
        out = out[~out.index.duplicated(keep="last")]
        
        return out

    macro: pd.DataFrame | None = None
    
    if cache_csv and os.path.exists(cache_csv) and not rebuild_cache:
        macro = pd.read_csv(cache_csv)
        if "public_date" not in macro.columns:
            date_candidates = [c for c in macro.columns if "date" in c.lower()]
            date_col = date_candidates[0] if date_candidates else macro.columns[0]
            macro["public_date"] = pd.to_datetime(macro[date_col], errors="coerce")
        else:
            macro["public_date"] = pd.to_datetime(macro["public_date"], errors="coerce")
        
        macro = macro.drop(
            columns=[c for c in ["Unnamed: 0", "index"] if c in macro.columns],
            errors="ignore",
        )
        macro = macro.dropna(subset=["public_date"]).copy()

    if macro is None:
        start = pd.to_datetime(panel["public_date"].min()) - pd.DateOffset(years=2)
        end = pd.to_datetime(panel["public_date"].max())

        print("Fetching FRED macro data...")
        fed = pdr.DataReader("FEDFUNDS", "fred", start, end)
        dgs10 = pdr.DataReader("DGS10", "fred", start, end)

        try:
            cpi = pdr.DataReader(cpi_series_prefer, "fred", start, end)
            cpi_name = cpi_series_prefer
        except Exception:
            cpi = pdr.DataReader("CPIAUCSL", "fred", start, end)
            cpi_name = "CPIAUCSL"

        gdp = pdr.DataReader("GDP", "fred", start, end)

        fed = _to_month_end_index(fed).rename(columns={"FEDFUNDS": "FEDFUNDS"}).ffill()
        dgs10 = _to_month_end_index(dgs10).rename(columns={"DGS10": "DGS10"}).ffill()
        cpi = _to_month_end_index(cpi).rename(columns={cpi_name: cpi_name}).ffill()
        gdp = _to_month_end_index(gdp).rename(columns={"GDP": "GDP"}).ffill()

        cpi_ser = cpi[cpi_name]
        gdp_ser = gdp["GDP"]

        # FIXED: Use vectorized pct_change instead of slow rolling apply
        inf_1mo = cpi_ser.pct_change(periods=1)
        inf_1yr = cpi_ser.pct_change(periods=12) # 12 months
        
        gdp_1mo = gdp_ser.pct_change(periods=1)
        gdp_1yr = gdp_ser.pct_change(periods=12) 

        macro = (
            pd.concat(
                [
                    fed,
                    cpi,
                    pd.DataFrame({"1mo_inf_rate": inf_1mo, "1yr_inf_rate": inf_1yr}),
                    gdp,
                    pd.DataFrame({"1mo_GDP": gdp_1mo, "1yr_GDP": gdp_1yr}),
                    dgs10,
                ],
                axis=1,
            )
            .reset_index()
            .rename(columns={"index": "public_date"}) # concat index name
        )
        
        # If reset_index name was not 'index', fix it
        if "public_date" not in macro.columns:
            # Usually concat index becomes 'index' or 'Date'
            macro = macro.rename(columns={"Date": "public_date", "index": "public_date"})

        macro['GDP'] = macro['GDP'].replace(0.0, np.nan).ffill()
        macro['1mo_GDP'] = macro['1mo_GDP'].replace(0.0, np.nan).ffill()
        macro['1yr_GDP'] = macro['1yr_GDP'].replace(0.0, np.nan).ffill()

        if cpi_name != "USACPIALLMINMEI":
            macro = macro.rename(columns={cpi_name: "USACPIALLMINMEI"})

        if cache_csv:
            os.makedirs(os.path.dirname(cache_csv) or ".", exist_ok=True)
            macro.to_csv(cache_csv, index=False)

    macro = macro.sort_values("public_date").copy()
    
    if shift_months:
        cols = ["FEDFUNDS", "USACPIALLMINMEI", "1mo_inf_rate", "1yr_inf_rate", 
                "GDP", "1mo_GDP", "1yr_GDP", "DGS10"]
        existing = [c for c in cols if c in macro.columns]
        macro[existing] = macro[existing].shift(shift_months)
        print(f"Applied {shift_months}-month lag to macro features")

    out = panel.merge(macro, on="public_date", how="left")
    return out


def main():
    ap = argparse.ArgumentParser(
        description="Build monthly panel for stock selection with improved paper compliance"
    )
    ap.add_argument("--cache-dir", default="data/simfin",
                   help="Directory with SimFin CSV cache files (default: data/simfin)")
    ap.add_argument("--out", default="data/simfin_panel.csv", 
                   help="Output CSV path")
    ap.add_argument("--add-macro", action="store_true", 
                   help="Fetch & merge macro series from FRED")
    ap.add_argument("--macro-cache", default="data/fred_macro_cache.csv", 
                   help="Where to cache FRED macro data")
    ap.add_argument("--macro-shift-months", type=int, default=1, 
                   help="Lag macro features by N months (default 1, paper recommendation)")
    ap.add_argument("--macro-rebuild", action="store_true", 
                   help="Rebuild macro cache even if cache CSV exists")
    ap.add_argument("--fundamental-lag-months", type=int, default=1,
                   help="Lag fundamental publish dates by N months for PIT alignment (default 1)")
    ap.add_argument("--exclude-financials", action="store_true", default=True,
                   help="Exclude financial companies (default True, standard practice)")
    ap.add_argument("--no-exclude-financials", dest="exclude_financials", 
                   action="store_false",
                   help="Do NOT exclude financial companies")
    ap.add_argument("--winsorize", action="store_true", default=True,
                   help="Winsorize extreme ratio values (default True)")
    ap.add_argument("--no-winsorize", dest="winsorize", action="store_false",
                   help="Do NOT winsorize ratios")
    ap.add_argument("--winsorize-mad", type=float, default=5.0,
                   help="Number of MADs for winsorization (default 5)")
    ap.add_argument("--apply-quality-filters", action="store_true", default=True,
                   help="Apply data quality filters (default True)")
    ap.add_argument("--no-quality-filters", dest="apply_quality_filters", 
                   action="store_false",
                   help="Do NOT apply quality filters")
    ap.add_argument("--year-adjusted-caps", action="store_true", default=True,
                   help="Use year-adjusted market cap categories (default True, paper methodology)")
    ap.add_argument("--static-caps", dest="year_adjusted_caps", action="store_false",
                   help="Use static market cap boundaries (not year-adjusted)")
    
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    print("="*80)
    print("BUILDING SIMFIN PANEL WITH PAPER COMPLIANCE IMPROVEMENTS")
    print("="*80)
    print(f"Fundamental lag: {args.fundamental_lag_months} months")
    print(f"Macro lag: {args.macro_shift_months} months")
    print(f"Exclude financials: {args.exclude_financials}")
    print(f"Winsorize ratios: {args.winsorize}")
    print(f"Quality filters: {args.apply_quality_filters}")
    print(f"Year-adjusted market caps: {args.year_adjusted_caps}")
    print("="*80 + "\n")

    # Build base datasets
    print("Building monthly prices...")
    monthly = build_monthly_prices(args.cache_dir)
    print(f"  {len(monthly)} monthly observations for {monthly['TICKER'].nunique()} tickers\n")
    
    print("Building quarterly fundamentals...")
    fund = build_quarterly_fundamentals(args.cache_dir)
    print(f"  {len(fund)} quarterly observations\n")
    
    # Merge with PIT alignment
    print("Merging monthly prices with fundamentals...")
    panel = merge_monthly_with_fundamentals(
        monthly, fund, 
        lag_months=args.fundamental_lag_months
    )
    print(f"  {len(panel)} rows after merge\n")

    # Exclude financials
    if args.exclude_financials:
        panel = exclude_financial_companies(panel, args.cache_dir)

    # Clean target variable
    panel["1yr_return"] = panel["1yr_return"].replace([np.inf, -np.inf], np.nan)

    # Apply data quality filters
    if args.apply_quality_filters:
        panel = apply_data_quality_filters(panel)

    # Apply year-adjusted market cap categories
    if args.year_adjusted_caps:
        panel = apply_year_adjusted_market_caps(panel)
        print()

    # Winsorize extreme ratios
    if args.winsorize:
        # Identify ratio columns
        ratio_keywords = ['ratio', 'turn', 'ptb', 'ps', 'pcf', 'pe_', 'bm', 
                         'npm', 'gpm', 'roa', 'roe', 'evm', 'dpr', 'opm', 
                         'ptpm', 'cfm', 'roce', 'efftax', 'debt_', 'int_',
                         'profit_', 'ocf_', 'cash_', 'fcf_', 'accrual', 
                         'sale_', 'rd_sale', 'lt_ppent']
        ratio_cols = [col for col in panel.columns 
                     if any(keyword in col.lower() for keyword in ratio_keywords)]
        panel = winsorize_ratios(panel, ratio_cols, n_mad=args.winsorize_mad)
        print()

    # Add macro features
    if args.add_macro:
        print("Adding FRED macro features...")
        panel = add_fred_macro(
            panel, 
            cache_csv=args.macro_cache, 
            rebuild_cache=args.macro_rebuild, 
            shift_months=args.macro_shift_months
        )
        print()

    # Drop rows lacking essentials
    panel = panel.dropna(subset=["gvkey","public_date","MthPrc","MthCap"]).copy()

    # Sort for consistency
    panel = panel.sort_values(["TICKER", "public_date"]).reset_index(drop=True)

    # Save output
    panel.to_csv(args.out, index=False)

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Output file: {args.out}")
    print(f"Shape: {panel.shape} ({panel.shape[0]} rows × {panel.shape[1]} columns)")
    print(f"Date range: {panel['public_date'].min()} → {panel['public_date'].max()}")
    print(f"Unique tickers: {panel['TICKER'].nunique()}")
    print(f"Rows with non-null 1yr_return: {panel['1yr_return'].notna().sum()}")
    print("\nMarket cap distribution:")
    print(panel['cap'].value_counts().sort_index())
    print("\nSample of features (first 20 columns):")
    print(panel.columns[:20].tolist())
    print("="*80)


if __name__ == "__main__":
    main()