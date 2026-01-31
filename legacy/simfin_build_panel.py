import os
import argparse
import numpy as np
import pandas as pd

# ------------------------------------------------------------
# SimFin -> monthly PIT panel for Long-Term-Stock-Selection
#
# Output columns (core):
#   TICKER, gvkey(=SimFinId), public_date (month-end)
#   MthPrc, Shares Outstanding, MthCap, divyield
#   1yr_return  (FORWARD 12-month return)
#
# Output columns (PIT fundamentals from quarterly statements via Publish Date):
#   Starter factors (v2): bm, ptb, ps, pcf, npm, gpm, roa, roe, de_ratio,
#                         curr_ratio, quick_ratio, cash_ratio
#   Extended factors (v3): evm, debt_ebitda, pe_inc, pe_exi, pe_op_basic, pe_op_dil,
#                          dpr, opmbd, opmad, ptpm, cfm, roce, efftax,
#                          capital_ratio, int_debt, int_totdebt, cash_lt,
#                          invt_act, rect_act, debt_at, short_debt, curr_debt, lt_debt,
#                          profit_lct, ocf_lct, cash_debt, fcf_ocf, lt_ppent, dltt_be,
#                          debt_assets, debt_capital, intcov, intcov_ratio,
#                          cash_conversion, inv_turn, at_turn, rect_turn, pay_turn,
#                          sale_invcap, sale_equity, sale_nwc, rd_sale, accrual, GProf
#
# Optional (free macro from FRED):
#   FEDFUNDS, DGS10, 1mo_inf_rate, 1yr_inf_rate, 1mo_GDP, 1yr_GDP
#
# Notes:
# - This script prefers correctness + PIT alignment over matching every WRDS naming nuance.
# - Some paper features (e.g., adv_sale, staff_sale) cannot be derived from these SimFin
#   statement files; they will not be produced here unless you add another data source.
# ------------------------------------------------------------

EPS = 1e-12

def _read_simfin(path: str, usecols=None) -> pd.DataFrame:
    return pd.read_csv(path, sep=";", usecols=usecols, low_memory=False)

def _to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _available_cols(path: str):
    # read header only
    return list(pd.read_csv(path, sep=";", nrows=0).columns)


def cap_bucket(mcap: float):
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
    px["public_date"] = px["Date"].dt.to_period("M").dt.to_timestamp("M")

    # month-end price & shares (last trading day in month)
    last_in_month = (
        px.drop_duplicates(["Ticker","public_date"], keep="last")
          .loc[:, ["Ticker","SimFinId","public_date","Adj. Close","Shares Outstanding"]]
          .rename(columns={"Adj. Close":"MthPrc"})
    )

    # trailing-12m dividends for dividend yield
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

    m = last_in_month.merge(div_m[["Ticker","public_date","Div_ttm"]], on=["Ticker","public_date"], how="left")
    m = m.sort_values(["Ticker","public_date"])

    # forward-fill shares inside ticker
    m["Shares Outstanding"] = m.groupby("Ticker")["Shares Outstanding"].ffill()

    m["MthCap"] = m["MthPrc"] * m["Shares Outstanding"]
    m["divyield"] = np.where(m["MthPrc"] > 0, m["Div_ttm"].fillna(0.0) / m["MthPrc"], np.nan)

    # Forward 12-month return target: (P_{t+12} - P_t) / P_t
    p = m.groupby("Ticker")["MthPrc"]
    fwd = p.shift(-12)
    m["1yr_return"] = (fwd - m["MthPrc"]) / (m["MthPrc"] + EPS)
    # guard against weird zero/negative prices
    m.loc[m["MthPrc"] <= 0, "1yr_return"] = np.nan

    # rename to repo-friendly identifiers
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
        # optional extras for later extensions (safe if absent)
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

    # numeric conversion
    for df in (inc, bal, cf):
        for c in df.columns:
            if c not in ("Ticker","Fiscal Year","Fiscal Period","Publish Date"):
                df[c] = _to_num(df[c])

    # merge by fiscal keys
    f = (
        inc.merge(bal, on=["Ticker","Fiscal Year","Fiscal Period","Publish Date"], how="outer")
           .merge(cf,  on=["Ticker","Fiscal Year","Fiscal Period","Publish Date"], how="outer")
           .sort_values(["Ticker","Publish Date"])
    )

    # trailing-4Q sums for flow variables
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
                 .rolling(4, min_periods=1).sum()
                 .reset_index(level=0, drop=True)
            )

    # debt helper
    if "Short Term Debt" in f.columns and "Long Term Debt" in f.columns:
        f["total_debt"] = f["Short Term Debt"].fillna(0) + f["Long Term Debt"].fillna(0)

    return f

def merge_monthly_with_fundamentals(monthly: pd.DataFrame, fundamentals: pd.DataFrame) -> pd.DataFrame:
    # merge_asof requires global sorting by "on" key (date) even when using by=...
    left = monthly.dropna(subset=["public_date","TICKER"]).copy()
    right = fundamentals.dropna(subset=["Publish Date","Ticker"]).copy()

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
        allow_exact_matches=True,
    )
    out = out.sort_values(["TICKER","public_date"]).reset_index(drop=True)

    eps = EPS

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

    # ----- Extended ratios (v3) -----
    out["cash"] = out["Cash, Cash Equivalents & Short Term Investments"]
    out["EBIT_ttm"] = out["Operating Income (Loss)_ttm"]
    out["EBITDA_ttm"] = out["Operating Income (Loss)_ttm"] + out["Depreciation & Amortization_ttm"]

    # EV and EV/EBITDA
    out["EV"] = out["MthCap"] + out["total_debt"] - out["cash"].fillna(0)
    out["evm"] = out["EV"] / (out["EBITDA_ttm"] + eps)
    out["debt_ebitda"] = out["total_debt"] / (out["EBITDA_ttm"] + eps)

    # P/E variants (cap-based)
    out["pe_inc"] = out["MthCap"] / (out["Net Income_ttm"] + eps)
    if "Net Income (Common)_ttm" in out.columns:
        out["pe_exi"] = out["MthCap"] / (out["Net Income (Common)_ttm"] + eps)
    else:
        out["pe_exi"] = out["pe_inc"]

    # operating earnings multiples (approximations)
    out["pe_op_basic"] = out["MthCap"] / (out["Operating Income (Loss)_ttm"] + eps)
    out["pe_op_dil"] = out["pe_op_basic"]

    # payout ratio: dividends / net income (Dividends Paid is usually negative)
    out["dpr"] = (-out["Dividends Paid_ttm"].fillna(0)) / (out["Net Income_ttm"] + eps)

    # margins
    out["opmbd"] = out["Operating Income (Loss)_ttm"] / (out["Revenue_ttm"] + eps)   # EBIT margin
    out["opmad"] = out["EBITDA_ttm"] / (out["Revenue_ttm"] + eps)                   # EBITDA margin
    out["ptpm"]  = out["Pretax Income (Loss)_ttm"] / (out["Revenue_ttm"] + eps)
    out["cfm"]   = out["Net Cash from Operating Activities_ttm"] / (out["Revenue_ttm"] + eps)

    # return on capital employed (approx)
    capital_employed = out["Total Assets"] - out["Total Current Liabilities"]
    out["roce"] = out["EBIT_ttm"] / (capital_employed + eps)

    # effective tax rate
    out["efftax"] = out["Income Tax (Expense) Benefit, Net_ttm"] / (out["Pretax Income (Loss)_ttm"] + eps)

    # capital structure / balance composition
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

    # interest coverage & interest-to-debt
    out["intcov"] = out["EBIT_ttm"] / (out["Interest Expense, Net_ttm"].abs() + eps)
    out["intcov_ratio"] = out["intcov"]
    out["int_debt"] = out["Interest Expense, Net_ttm"].abs() / (out["total_debt"] + eps)
    out["int_totdebt"] = out["int_debt"]

    # profitability vs current liabilities
    out["profit_lct"] = out["Net Income_ttm"] / (out["Total Current Liabilities"] + eps)
    out["ocf_lct"]    = out["Net Cash from Operating Activities_ttm"] / (out["Total Current Liabilities"] + eps)

    # cash / debt ratios
    out["cash_debt"] = out["cash"] / (out["total_debt"] + eps)
    out["cash_conversion"] = out["Net Cash from Operating Activities_ttm"] / (out["Net Income_ttm"] + eps)

    # capex proxy + FCF ratios
    capex = (-out["Change in Fixed Assets & Intangibles_ttm"]).clip(lower=0)
    out["capex_ttm"] = capex
    out["fcf_ttm"] = out["Net Cash from Operating Activities_ttm"] - out["capex_ttm"]
    out["fcf_ocf"] = out["fcf_ttm"] / (out["Net Cash from Operating Activities_ttm"] + eps)

    # accruals (Sloan-style)
    out["accrual"] = (out["Net Income_ttm"] - out["Net Cash from Operating Activities_ttm"]) / (out["Total Assets"] + eps)

    # gross profitability (asset-pricing standard)
    out["GProf"] = out["Gross Profit_ttm"] / (out["Total Assets"] + eps)

    # turnover ratios
    out["inv_turn"]  = out["Cost of Revenue_ttm"] / (out["Inventories"] + eps)
    out["rect_turn"] = out["Revenue_ttm"] / (out["Accounts & Notes Receivable"] + eps)
    out["pay_turn"]  = out["Cost of Revenue_ttm"] / (out["Payables & Accruals"] + eps)
    out["at_turn"]   = out["Revenue_ttm"] / (out["Total Assets"] + eps)

    # sales efficiency / invested capital
    out["invcap"] = out["Total Equity"] + out["total_debt"] - out["cash"].fillna(0)
    out["sale_invcap"] = out["Revenue_ttm"] / (out["invcap"] + eps)
    out["sale_equity"] = out["Revenue_ttm"] / (out["Total Equity"] + eps)
    out["sale_nwc"] = out["Revenue_ttm"] / ((out["Total Current Assets"] - out["Total Current Liabilities"]) + eps)

    # R&D intensity
    out["rd_sale"] = out["Research & Development_ttm"] / (out["Revenue_ttm"] + eps)

    # PPE intensity
    out["lt_ppent"] = out["Property, Plant & Equipment, Net"] / (out["Total Assets"] + eps)

    # cap bucket
    out["cap"] = out["MthCap"].apply(cap_bucket)

    # cleanup
    out = out.drop(columns=["Ticker"], errors="ignore")
    return out

def add_fred_macro(
    panel: pd.DataFrame,
    cache_csv: str | None = None,
    rebuild_cache: bool = False,
    shift_months: int = 0,
    cpi_series_prefer: str = "USACPIALLMINMEI",
) -> pd.DataFrame:
    """
    Add macro series from FRED and merge on month-end `public_date`.

    This follows the *same style* used in the original project's Cleaning Notebook:
      - GDP is quarterly, forward-filled to monthly and then "1mo" / "1yr" changes
        are computed via rolling-window pct-change over 2 / 13 months.
      - Inflation changes are computed the same way.
      - DGS10 is treated as daily and downsampled to month-end (last obs in month).
      - FEDFUNDS is treated as monthly and aligned to month-end.

    Notes
    -----
    - We intentionally fetch each series separately. Fetching multiple FRED series
      in one request aligns them to a union calendar (often daily), which can
      make month-end "last" land on NaNs for monthly series (the issue you saw).
    - `shift_months` can be used as a conservative lag (default 0 to match the
      notebook; set to 1 if you want to reduce lookahead risk).
    """
    try:
        from pandas_datareader import data as pdr
    except Exception as e:
        raise RuntimeError(
            "pandas_datareader is required for --add-macro (pip install pandas_datareader)"
        ) from e

    # --------------------------
    # Helpers
    # --------------------------
    def _to_month_end_index(df: pd.DataFrame) -> pd.DataFrame:
        # FRED comes with a DateTimeIndex; we standardize to month-end timestamps.
        out = df.copy()
        out.index = pd.to_datetime(out.index)
        out.index = out.index.to_period("M").to_timestamp("M")  # month-end
        out = out.sort_index()
        # If multiple rows collapse to same month-end, keep last
        out = out[~out.index.duplicated(keep="last")]
        return out

    def _rolling_pct_change(s: pd.Series, window: int) -> pd.Series:
        # Match Cleaning Notebook: (last - first)/first over the window
        def pct_change_window(x: pd.Series) -> float:
            # x is a Series with index positions 0..window-1
            first = x.iloc[0]
            last = x.iloc[-1]
            if pd.isna(first) or pd.isna(last) or first == 0:
                return np.nan
            return (last - first) / first

        return s.rolling(window).apply(pct_change_window, raw=False)

    # --------------------------
    # Load from cache if possible
    # --------------------------
    macro: pd.DataFrame | None = None
    if cache_csv and os.path.exists(cache_csv) and not rebuild_cache:
        macro = pd.read_csv(cache_csv)

        # Robust date column handling: cache may store the date as 'public_date', 'DATE',
        # or as the first column.
        if "public_date" not in macro.columns:
            date_candidates = [c for c in macro.columns if "date" in c.lower()]
            date_col = date_candidates[0] if date_candidates else macro.columns[0]
            macro["public_date"] = pd.to_datetime(macro[date_col], errors="coerce")
        else:
            macro["public_date"] = pd.to_datetime(macro["public_date"], errors="coerce")

        # Drop obvious junk index columns if present
        macro = macro.drop(
            columns=[c for c in ["Unnamed: 0", "index"] if c in macro.columns],
            errors="ignore",
        )

        # Remove rows without a valid date
        macro = macro.dropna(subset=["public_date"]).copy()

    if macro is None:
        start = pd.to_datetime(panel["public_date"].min()) - pd.DateOffset(years=2)
        end = pd.to_datetime(panel["public_date"].max())

        # Fetch each series independently (prevents NaN month-end selection issues)
        fed = pdr.DataReader("FEDFUNDS", "fred", start, end)
        dgs10 = pdr.DataReader("DGS10", "fred", start, end)

        # Prefer the series used in the original notebook if available
        try:
            cpi = pdr.DataReader(cpi_series_prefer, "fred", start, end)
            cpi_name = cpi_series_prefer
        except Exception:
            cpi = pdr.DataReader("CPIAUCSL", "fred", start, end)
            cpi_name = "CPIAUCSL"

        gdp = pdr.DataReader("GDP", "fred", start, end)

        # Align each to month-end
        fed = _to_month_end_index(fed).rename(columns={"FEDFUNDS": "FEDFUNDS"}).ffill()
        dgs10 = _to_month_end_index(dgs10).rename(columns={"DGS10": "DGS10"}).ffill()
        cpi = _to_month_end_index(cpi).rename(columns={cpi_name: cpi_name}).ffill()

        # GDP is quarterly -> month-end with forward fill (as per notebook)
        gdp = _to_month_end_index(gdp).rename(columns={"GDP": "GDP"}).ffill()

        # Build derived rates to match the notebook logic
        cpi_ser = cpi[cpi_name]
        gdp_ser = gdp["GDP"]

        inf_1mo = _rolling_pct_change(cpi_ser, 2)
        inf_1yr = _rolling_pct_change(cpi_ser, 13)

        gdp_1mo = _rolling_pct_change(gdp_ser, 2)
        gdp_1yr = _rolling_pct_change(gdp_ser, 13)

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
            .rename(columns={"index": "public_date"})
        )

        # Keep a stable column name for CPI (the notebook expects USACPIALLMINMEI)
        if cpi_name != "USACPIALLMINMEI":
            macro = macro.rename(columns={cpi_name: "USACPIALLMINMEI"})

        # Optional cache
        if cache_csv:
            os.makedirs(os.path.dirname(cache_csv) or ".", exist_ok=True)
            macro.to_csv(cache_csv, index=False)

    # Optional lag to reduce lookahead risk
    macro = macro.sort_values("public_date").copy()
    if shift_months:
        cols = ["FEDFUNDS", "USACPIALLMINMEI", "1mo_inf_rate", "1yr_inf_rate", "GDP", "1mo_GDP", "1yr_GDP", "DGS10"]
        existing = [c for c in cols if c in macro.columns]
        macro[existing] = macro[existing].shift(shift_months)

    out = panel.merge(macro, on="public_date", how="left")
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default="data/simfin", help="Directory with SimFin CSV cache files (default: data/simfin)")
    ap.add_argument("--out", default="data/simfin_panel.csv", help="Output CSV path (default: data/simfin_panel.csv)")
    ap.add_argument("--add-macro", action="store_true", help="Fetch & merge macro series from FRED")
    ap.add_argument("--macro-cache", default="data/fred_macro_cache.csv", help="Where to cache FRED macro data")
    ap.add_argument("--macro-shift-months", type=int, default=0, help="Lag macro features by N months (default 0)")
    ap.add_argument("--macro-rebuild", action="store_true", help="Rebuild macro cache even if cache CSV exists")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    monthly = build_monthly_prices(args.cache_dir)
    fund = build_quarterly_fundamentals(args.cache_dir)
    panel = merge_monthly_with_fundamentals(monthly, fund)

    # clean target
    panel["1yr_return"] = panel["1yr_return"].replace([np.inf, -np.inf], np.nan)

    if args.add_macro:
        panel = add_fred_macro(panel, cache_csv=args.macro_cache, rebuild_cache=args.macro_rebuild, shift_months=args.macro_shift_months)

    # drop rows lacking essentials (keep X missing; handle in model via imputation)
    panel = panel.dropna(subset=["gvkey","public_date","MthPrc","MthCap"]).copy()

    panel.to_csv(args.out, index=False)

    print(f"Wrote {args.out} shape= {panel.shape} cols= {len(panel.columns)}")
    print("Date range:", panel["public_date"].min(), "->", panel["public_date"].max())
    print("Non-null target rows:", int(panel["1yr_return"].notna().sum()))

if __name__ == "__main__":
    main()
