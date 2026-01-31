import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import calendar
from sklearn.ensemble import RandomForestRegressor

warnings.filterwarnings('ignore')

# ==============================================================================
# PART 1: MODEL CLASS (Unchanged)
# ==============================================================================

class StockSelectionRF:
    def __init__(self, n_estimators=500, max_depth=12, random_state=42):
        self.model = RandomForestRegressor(
            n_estimators=n_estimators, max_depth=max_depth, max_features=0.3,
            min_samples_leaf=50, max_samples=0.7, random_state=random_state, n_jobs=-1
        )
        self.feature_medians_ = None
        
    def prepare_features(self, df):
        valuation = ['bm', 'ptb', 'ps', 'pcf', 'pe_inc', 'pe_exi', 'pe_op_basic', 'pe_op_dil', 'evm', 'dpr', 'divyield']
        profitability = ['npm', 'gpm', 'roa', 'roe', 'roce', 'opmbd', 'opmad', 'ptpm', 'cfm', 'efftax', 'GProf']
        solvency = ['de_ratio', 'debt_at', 'debt_assets', 'debt_capital', 'capital_ratio', 'intcov', 'intcov_ratio', 'dltt_be']
        liquidity = ['curr_ratio', 'quick_ratio', 'cash_ratio', 'cash_conversion']
        efficiency = ['at_turn', 'inv_turn', 'rect_turn', 'pay_turn', 'sale_invcap', 'sale_equity', 'sale_nwc']
        financial_soundness = ['cash_lt', 'invt_act', 'rect_act', 'short_debt', 'curr_debt', 'lt_debt', 'profit_lct', 'ocf_lct', 'cash_debt', 'fcf_ocf']
        other = ['accrual', 'rd_sale', 'lt_ppent']
        macro = ['FEDFUNDS', 'DGS10', '1mo_inf_rate', '1yr_inf_rate', '1mo_GDP', '1yr_GDP']
        size = ['MthCap']
        
        all_features = (valuation + profitability + solvency + liquidity + efficiency + financial_soundness + other + macro + size)
        return [col for col in all_features if col in df.columns]
    
    def neutralize_features(self, X, metadata):
        if 'sector' not in metadata.columns: return X
        X_neutral = X.copy()
        combined = X.copy()
        combined['sector'] = metadata['sector']
        combined['public_date'] = metadata['public_date']
        
        for col in X.columns:
            if col in ['sector', 'public_date']: continue
            grouped = combined.groupby(['sector', 'public_date'])[col]
            group_mean = grouped.transform('mean')
            group_std = grouped.transform('std')
            X_neutral[col] = (combined[col] - group_mean) / (group_std + 1e-8)
            X_neutral[col] = X_neutral[col].replace([np.inf, -np.inf], np.nan).fillna(0)
        return X_neutral
    
    def winsorize_target(self, y, lower_pct=1, upper_pct=99):
        lower = np.percentile(y.dropna(), lower_pct)
        upper = np.percentile(y.dropna(), upper_pct)
        return y.clip(lower=lower, upper=upper)
    
    def rank_target(self, y, dates):
        temp = pd.DataFrame({'target': y, 'date': dates})
        temp['rank'] = temp.groupby('date')['target'].rank(pct=True)
        return temp['rank']
    
    def handle_missing_data(self, X, missing_threshold_pct=0.5):
        missing_pct = X.isnull().sum() / len(X)
        feats_keep = missing_pct[missing_pct <= missing_threshold_pct].index.tolist()
        dropped_feats = missing_pct[missing_pct > missing_threshold_pct].index.tolist()
        
        X_filt = X[feats_keep].copy()
        row_comp = X_filt.notna().sum(axis=1) / X_filt.shape[1]
        X_filt = X_filt[row_comp >= 0.8].copy()
        
        if not hasattr(self, 'feature_medians_') or self.feature_medians_ is None:
            self.feature_medians_ = X_filt.median()
        
        X_clean = X_filt.fillna(self.feature_medians_).fillna(0)
        return X_clean, dropped_feats

    def train(self, X_train, y_train, metadata_train):
        valid = y_train.notna()
        X_train, y_train = X_train[valid], y_train[valid]
        metadata_train = metadata_train[valid] if metadata_train is not None else None
        
        X_neut = self.neutralize_features(X_train, metadata_train)
        y_wins = self.winsorize_target(y_train)
        y_rank = self.rank_target(y_wins, metadata_train['public_date'])
        
        self.model.fit(X_neut, y_rank)
        
    def get_top_stocks(self, X_test, test_idx, metadata_test, n=10):
        X_neut = self.neutralize_features(X_test, metadata_test)
        preds = self.model.predict(X_neut)
        res = pd.DataFrame({'predicted_rank': preds}, index=test_idx)
        return res.nlargest(n, 'predicted_rank')

# ==============================================================================
# PART 2: DATA PREP & 3-MONTH RETURN CALCULATION
# ==============================================================================

def load_and_prepare_data(filepath):
    print(f"Loading data from {filepath}...")
    try:
        df = pd.read_csv(filepath)
    except:
        df = pd.read_csv(filepath, sep=';')
        
    # 1. Date Snapping
    date_col = next((c for c in ['public_date', 'Date', 'Report Date'] if c in df.columns), None)
    if not date_col: raise ValueError("No date column found")
    
    df.rename(columns={date_col: 'public_date'}, inplace=True)
    df['public_date'] = pd.to_datetime(df['public_date']) + pd.offsets.MonthEnd(0)
    
    # 2. Ticker Standardizing
    if 'Ticker' in df.columns and 'TICKER' not in df.columns:
        df.rename(columns={'Ticker': 'TICKER'}, inplace=True)
        
    df = df.sort_values(['TICKER', 'public_date'])

    # 3. CALCULATE 3-MONTH TARGET
    price_cols = ['Adj. Close', 'Adj Close', 'MthPrc', 'Close']
    price_col = next((c for c in price_cols if c in df.columns), None)
    
    if price_col:
        print(f"Calculating 3-Month Returns using '{price_col}'...")
        # Shift(-3) gets the price 3 months ahead
        df['fut_3m_ret'] = df.groupby('TICKER')[price_col].shift(-3) / df[price_col] - 1
        
        # Verify gaps
        future_dates = df.groupby('TICKER')['public_date'].shift(-3)
        days_diff = (future_dates - df['public_date']).dt.days
        valid_gap = (days_diff >= 80) & (days_diff <= 105)
        df.loc[~valid_gap, 'fut_3m_ret'] = np.nan
        
        # --- NEW: Calculate Outcome Realization Date ---
        # This is the date the return was officially "known" in the real world
        df['outcome_date'] = df['public_date'] + pd.DateOffset(months=3)
        
        print(f"  Target calculated. Valid rows: {df['fut_3m_ret'].notna().sum()}")
    else:
        print("  [Error] No price column found. Cannot calculate 3m target.")
        exit()

    return df

def infer_sector_from_data(df):
    if 'sector' in df.columns: return df
    if 'industry' in df.columns:
        df['sector'] = df['industry']
        return df
    df['size'] = pd.qcut(df.groupby('TICKER')['MthCap'].transform('median'), 3, labels=['S','M','L'])
    df['prof'] = pd.qcut(df.groupby('TICKER')['roe'].transform('median'), 3, labels=['L','M','H'])
    df['sector'] = df['size'].astype(str) + '-' + df['prof'].astype(str)
    return df

# ==============================================================================
# PART 3: STRICT POINT-IN-TIME SELECTION (THE FIX)
# ==============================================================================

def select_stocks_pit(df_full, target_date_str, n_stocks=10, min_cap='Mid Cap'):
    target_dt = pd.to_datetime(target_date_str)
    
    # 1. Cap Filtering
    cap_hierarchy = ['Nano Cap', 'Micro Cap', 'Small Cap', 'Mid Cap', 'Large Cap', 'Mega Cap']
    df = df_full.copy()
    df = infer_sector_from_data(df)
    
    if 'cap' in df.columns:
        df['cap'] = df['cap'].astype(str).str.strip()
        try:
            min_cap_idx = cap_hierarchy.index(min_cap)
            allowed_caps = cap_hierarchy[min_cap_idx:]
            df = df[df['cap'].isin(allowed_caps)].copy()
        except ValueError:
            pass # Use all if cap not found
    
    # 2. STRICT TRAIN/TEST SPLIT
    # -------------------------------------------------------------------------
    # Test Data: The features available AT target_dt
    test_mask = df['public_date'] == target_dt
    test_df = df[test_mask].copy()
    
    # Train Data: THE CRITICAL FIX
    # We can only train on rows where the outcome (3-month return) 
    # was ALREADY KNOWN before the target_dt.
    # Logic: outcome_date < target_dt
    
    # We add a small buffer (e.g., 1 day) just to be absolutely sure we don't 
    # train on something that finished *today*.
    
    train_mask = (df['outcome_date'] < target_dt) & (df['fut_3m_ret'].notna())
    train_df = df[train_mask].copy()
    
    if len(test_df) < n_stocks or len(train_df) < 500:
        return []
        
    # 3. RUN MODEL
    rf = StockSelectionRF()
    feature_cols = rf.prepare_features(df)
    
    X_train = train_df[feature_cols].copy()
    X_train_clean, dropped = rf.handle_missing_data(X_train)
    valid_feats = [f for f in feature_cols if f not in dropped]
    
    X_test = test_df[valid_feats].copy()
    X_test_clean = X_test.fillna(rf.feature_medians_).fillna(0)
    
    y_train = train_df.loc[X_train_clean.index, 'fut_3m_ret']
    
    meta_train = train_df.loc[X_train_clean.index, ['sector', 'public_date']]
    meta_test = test_df.loc[X_test_clean.index, ['sector', 'public_date']]
    
    rf.train(X_train_clean, y_train, meta_train)
    top_stocks = rf.get_top_stocks(X_test_clean, X_test_clean.index, meta_test, n=n_stocks)
    
    return top_stocks.index.tolist()

# ==============================================================================
# PART 4: EXECUTION
# ==============================================================================

def get_return_from_csv(df_full, stock_indices, start_date, end_date):
    if not stock_indices: return 0.0
    
    price_cols = ['Adj. Close', 'Adj Close', 'MthPrc', 'Close']
    price_col = next((c for c in price_cols if c in df_full.columns), None)
    
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    
    buy_rows = df_full.loc[stock_indices]
    tickers = buy_rows['TICKER'].unique()
    
    total_ret = 0.0
    count = 0
    
    for ticker in tickers:
        try:
            buy_price = buy_rows[buy_rows['TICKER'] == ticker][price_col].iloc[0]
        except: continue
        
        # Find Sell Price
        sell_row = df_full[
            (df_full['TICKER'] == ticker) & 
            (df_full['public_date'] >= end_dt - timedelta(days=5)) &
            (df_full['public_date'] <= end_dt + timedelta(days=5))
        ]
        
        if sell_row.empty: continue
            
        sell_price = sell_row[price_col].iloc[0]
        
        if buy_price > 0:
            ret = (sell_price - buy_price) / buy_price
            total_ret += ret
            count += 1
            
    return total_ret / count if count > 0 else 0.0

def run_strategy():
    csv_path = 'data/simfin_panel.csv'
    df_full = load_and_prepare_data(csv_path)
    
    START_YEAR = 2017
    END_YEAR = 2025
    MONTHS = [2, 5, 8, 11]
    INIT_CASH = 10000
    
    dates = []
    for y in range(START_YEAR, END_YEAR + 1):
        for m in MONTHS:
            last_day = calendar.monthrange(y, m)[1]
            date_str = f"{y}-{m:02d}-{last_day}"
            dates.append(date_str)
            
    dates.append((pd.to_datetime(dates[-1]) + pd.DateOffset(months=3)).strftime('%Y-%m-%d'))
    
    print(f"{'='*60}")
    print(f"STRICT POINT-IN-TIME BACKTEST")
    print(f"Training Filter: outcome_date < trade_date")
    print(f"{'='*60}\n")
    
    capital = INIT_CASH
    history = []
    
    for i in range(len(dates)-1):
        buy_date = dates[i]
        sell_date = dates[i+1]
        
        print(f">>> {buy_date} to {sell_date}")
        
        # 1. SELECT (Using Strict PIT)
        stock_indices = select_stocks_pit(df_full, buy_date, n_stocks=10, min_cap='Mid Cap')
        
        if not stock_indices:
            print("  [Warn] No stocks selected. Cash.")
            port_ret = 0.0
        else:
            # 2. REALIZED RETURN
            port_ret = get_return_from_csv(df_full, stock_indices, buy_date, sell_date)
            
        capital *= (1 + port_ret)
        print(f"  Ret: {port_ret*100:+.2f}% | Bal: ${capital:,.0f}")
        
        history.append({
            'Start': buy_date, 'End': sell_date,
            'Balance': capital, 'Return': port_ret
        })
        
    pd.DataFrame(history).to_excel("results/backtest/Quarterly_Strict_PIT_Results.xlsx", index=False)
    print("\nDone.")

if __name__ == "__main__":
    run_strategy()