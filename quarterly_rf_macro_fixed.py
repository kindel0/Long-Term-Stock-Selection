import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import warnings
import os
from datetime import timedelta

warnings.filterwarnings('ignore')

# ==============================================================================
# 1. CORE RANDOM FOREST CLASS
# ==============================================================================
class StockSelectionRF:
    def __init__(self, n_estimators=200, max_depth=12, random_state=42):
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,      
            max_depth=max_depth,             
            max_features=0.3,                
            min_samples_leaf=50,             
            max_samples=0.7,                 
            random_state=random_state,
            n_jobs=-1
        )
        self.feature_columns = None
        self.feature_medians_ = None
        self.macro_columns = ['FEDFUNDS', 'DGS10', '1mo_inf_rate', '1yr_inf_rate', '1mo_GDP', '1yr_GDP']
        
    def prepare_features(self, df):
        # 1. Fundamental Ratios
        valuation = ['bm', 'ptb', 'ps', 'pcf', 'pe_inc', 'pe_exi', 'pe_op_basic', 'evm', 'divyield']
        profitability = ['npm', 'gpm', 'roa', 'roe', 'roce', 'opmbd', 'opmad', 'cfm', 'ptpm']
        solvency = ['de_ratio', 'debt_at', 'debt_capital', 'intcov_ratio', 'capital_ratio']
        liquidity = ['curr_ratio', 'quick_ratio', 'cash_conversion', 'cash_ratio']
        efficiency = ['at_turn', 'inv_turn', 'sale_invcap', 'rect_turn']
        size = ['MthCap']
        
        # 2. Macro Data (Global Economic Context)
        macro = [c for c in self.macro_columns if c in df.columns]
        
        all_features = (valuation + profitability + solvency + liquidity + 
                       efficiency + size + macro)
        return [col for col in all_features if col in df.columns]
    
    def neutralize_features(self, X, metadata):
        """
        Z-Score stock fundamentals relative to sector peers.
        Pass macro data through RAW (it's global context, not stock-specific).
        """
        if 'sector' not in metadata.columns:
            return X
            
        X_neutral = X.copy()
        combined = X.copy()
        combined['sector'] = metadata['sector'].values
        combined['public_date'] = metadata['public_date'].values
        
        for col in X.columns:
            if col in ['sector', 'public_date']:
                continue
            
            # CRITICAL FIX: Do NOT neutralize macro columns
            # Macro data is the same for all stocks - neutralizing it makes it useless
            if col in self.macro_columns:
                # Keep macro data as-is (raw values)
                continue

            # For stock fundamentals: Z-score relative to sector peers
            grouped = combined.groupby(['sector', 'public_date'])[col]
            mean = grouped.transform('mean')
            std = grouped.transform('std')
            X_neutral[col] = (combined[col] - mean) / (std + 1e-8)
            X_neutral[col] = X_neutral[col].replace([np.inf, -np.inf], np.nan).fillna(0)
            
        return X_neutral
    
    def winsorize_target(self, y, lower_pct=1, upper_pct=99):
        """Remove extreme outliers"""
        lower = np.percentile(y.dropna(), lower_pct)
        upper = np.percentile(y.dropna(), upper_pct)
        return y.clip(lower=lower, upper=upper)
    
    def rank_target(self, y, dates):
        """Convert returns to cross-sectional ranks (removes market beta)"""
        temp = pd.DataFrame({'y': y, 'd': dates})
        return temp.groupby('d')['y'].rank(pct=True)

    def handle_missing_data(self, X, missing_threshold_pct=0.5):
        """Handle missing values with median imputation"""
        # Drop features with too much missing data
        missing_pct = X.isnull().sum() / len(X)
        features_to_keep = missing_pct[missing_pct <= missing_threshold_pct].index.tolist()
        dropped = [c for c in X.columns if c not in features_to_keep]
        
        X_filtered = X[features_to_keep].copy()
        
        # Drop rows with too much missing data
        row_completeness = X_filtered.notna().sum(axis=1) / X_filtered.shape[1]
        X_filtered = X_filtered[row_completeness >= 0.8].copy()
        
        # Calculate medians for imputation (only on training data)
        if not hasattr(self, 'feature_medians_') or self.feature_medians_ is None:
            self.feature_medians_ = X_filtered.median()
        
        # Impute
        X_clean = X_filtered.fillna(self.feature_medians_).fillna(0)
        
        return X_clean, dropped

    def train(self, X_train, y_train, metadata_train):
        # Filter valid targets
        valid = y_train.notna()
        X_train = X_train[valid]
        y_train = y_train[valid]
        if metadata_train is not None:
            metadata_train = metadata_train[valid]
        
        print(f"  Training on {len(X_train)} samples with {X_train.shape[1]} features...")
        
        # Neutralize fundamentals, keep macro raw
        X_neutral = self.neutralize_features(X_train, metadata_train)
        
        # Winsorize and rank targets
        y_wins = self.winsorize_target(y_train)
        y_ranked = self.rank_target(y_wins, metadata_train['public_date'])
        
        self.feature_columns = X_neutral.columns.tolist()
        self.model.fit(X_neutral, y_ranked)
        
        # Report macro importance
        imps = pd.Series(self.model.feature_importances_, 
                        index=self.feature_columns).sort_values(ascending=False)
        
        # Find top macro features
        macro_imps = [(c, imps[c]) for c in imps.index if c in self.macro_columns]
        if macro_imps:
            top_macro = macro_imps[0]
            rank = list(imps.index).index(top_macro[0]) + 1
            print(f"  Top Macro: {top_macro[0]} (Rank #{rank}, Importance: {top_macro[1]:.4f})")
        else:
            print("  [Warn] No macro features used")

    def get_top_stocks(self, X_test, test_indices, metadata_test, n=25):
        X_neutral = self.neutralize_features(X_test, metadata_test)
        
        # Align columns with training
        if self.feature_columns:
            for c in self.feature_columns:
                if c not in X_neutral.columns:
                    X_neutral[c] = 0
            X_neutral = X_neutral[self.feature_columns]
        
        preds = self.model.predict(X_neutral)
        results = pd.DataFrame({'score': preds}, index=test_indices)
        return results.nlargest(n, 'score')

# ==============================================================================
# 2. DATA PREPARATION (TTM + MACRO + STRICT PIT)
# ==============================================================================
def load_and_prepare_data(filepath):
    print("Loading data...")
    df = pd.read_csv(filepath)
    
    print(f"  Initial shape: {df.shape}")
    
    # 1. Standardize dates
    df['public_date'] = pd.to_datetime(df['public_date']) + pd.offsets.MonthEnd(0)
    df = df.sort_values(['TICKER', 'public_date'])
    
    # 2. Detect frequency
    print("  > Detecting frequency...")
    sample_diffs = []
    for ticker in df['TICKER'].unique()[:10]:
        ticker_df = df[df['TICKER'] == ticker]
        if len(ticker_df) > 1:
            sample_diffs.append(ticker_df['public_date'].diff().median().days)
    
    avg_diff = np.median(sample_diffs) if sample_diffs else 30
    is_monthly = avg_diff < 45
    shift_periods = -3 if is_monthly else -1
    print(f"  Detected: {'Monthly' if is_monthly else 'Quarterly'} (avg: {avg_diff:.0f} days)")

    # 3. Calculate TTM Flows (Seasonality Fix)
    print("  > Calculating TTM flows...")
    flow_map = {
        'Revenue': ['Revenue', 'Sales'],
        'Net Income': ['Net Income', 'Net_Income'],
        'Gross Profit': ['Gross Profit', 'Gross_Profit']
    }
    
    ttm_count = 0
    for std_name, candidates in flow_map.items():
        raw_col = None
        for candidate in candidates:
            if candidate in df.columns:
                raw_col = candidate
                break
        
        if raw_col:
            df[f'{std_name}_TTM'] = df.groupby('TICKER')[raw_col].transform(
                lambda x: x.rolling(window=4, min_periods=4).sum()
            )
            ttm_count += 1
            print(f"    Calculated {std_name}_TTM from '{raw_col}'")
    
    print(f"  Successfully calculated {ttm_count} TTM variables")

    # 4. Re-calculate ratios using TTM
    print("  > Re-calculating ratios with TTM...")
    
    def safe_div(a, b):
        return np.where(b == 0, np.nan, a / b)
    
    # Find equity/asset columns
    equity_col = next((c for c in ['Total Equity', 'Equity'] if c in df.columns), None)
    assets_col = next((c for c in ['Total Assets', 'Assets'] if c in df.columns), None)
    
    # Update ratios
    if 'Net Income_TTM' in df.columns and 'MthCap' in df.columns:
        df['pe_inc'] = safe_div(df['MthCap'], df['Net Income_TTM'])
        print("    Updated pe_inc")
        
    if 'Revenue_TTM' in df.columns and 'MthCap' in df.columns:
        df['ps'] = safe_div(df['MthCap'], df['Revenue_TTM'])
        print("    Updated ps")
        
    if 'Net Income_TTM' in df.columns and equity_col:
        df['roe'] = safe_div(df['Net Income_TTM'], df[equity_col])
        print("    Updated roe")
        
    if 'Net Income_TTM' in df.columns and assets_col:
        df['roa'] = safe_div(df['Net Income_TTM'], df[assets_col])
        print("    Updated roa")
        
    if 'Net Income_TTM' in df.columns and 'Revenue_TTM' in df.columns:
        df['npm'] = safe_div(df['Net Income_TTM'], df['Revenue_TTM'])
        print("    Updated npm")
        
    if 'Gross Profit_TTM' in df.columns and 'Revenue_TTM' in df.columns:
        df['gpm'] = safe_div(df['Gross Profit_TTM'], df['Revenue_TTM'])
        print("    Updated gpm")

    # 5. Calculate 3-month forward returns with validation
    print("  > Calculating 3-month targets...")
    
    price_col = next((c for c in ['MthPrc', 'Adj. Close', 'Close'] if c in df.columns), None)
    if not price_col:
        raise ValueError("No price column found")
    
    print(f"    Using price: '{price_col}'")
    
    # Calculate next price and validate gap
    df['next_price'] = df.groupby('TICKER')[price_col].shift(shift_periods)
    df['next_date'] = df.groupby('TICKER')['public_date'].shift(shift_periods)
    df['days_gap'] = (df['next_date'] - df['public_date']).dt.days
    
    # Only keep returns with valid ~3-month gaps
    valid_gap = (df['days_gap'] >= 80) & (df['days_gap'] <= 105)
    df['3mo_return'] = np.where(
        valid_gap,
        (df['next_price'] - df[price_col]) / df[price_col],
        np.nan
    )
    
    valid_count = df['3mo_return'].notna().sum()
    print(f"    Valid 3-month returns: {valid_count} ({valid_count/len(df)*100:.1f}%)")
    
    # Clean up temp columns
    df.drop(['next_price', 'next_date', 'days_gap'], axis=1, inplace=True)
    
    # 6. Strict PIT outcome date (with reporting lag)
    print("  > Setting outcome dates...")
    df['outcome_date'] = df['public_date'] + pd.DateOffset(months=3) + pd.DateOffset(days=3)
    
    # 7. Infer sectors if missing
    if 'sector' not in df.columns:
        print("  > Creating pseudo-sectors...")
        df['size_q'] = pd.qcut(
            df.groupby('TICKER')['MthCap'].transform('median'),
            q=3, labels=['Small', 'Mid', 'Large'], duplicates='drop'
        )
        df['prof_q'] = pd.qcut(
            df.groupby('TICKER')['roe'].transform('median'),
            q=3, labels=['Low', 'Med', 'High'], duplicates='drop'
        )
        df['sector'] = df['size_q'].astype(str) + '_' + df['prof_q'].astype(str)
    
    print(f"  Final shape: {df.shape}")
    return df

# ==============================================================================
# 3. QUARTERLY BACKTEST (MACRO-AWARE)
# ==============================================================================
def quarterly_backtest_internal_macro(df, portfolio_size=25, start_year=2018, 
                                     min_market_cap=200e6):
    
    # Filter for liquidity
    print(f"\nFiltering for market cap > ${min_market_cap/1e6:.0f}M...")
    initial_len = len(df)
    df = df[df['MthCap'] > min_market_cap].copy()
    print(f"  Kept {len(df)} / {initial_len} rows ({len(df)/initial_len*100:.1f}%)")
    
    # Generate test dates (quarterly)
    all_dates = sorted(df['public_date'].unique())
    test_dates = [d for d in all_dates 
                  if d.year >= start_year and d.month in [3, 6, 9, 12]]
    
    print(f"\nBacktest: {len(test_dates)} periods from {test_dates[0].date()} to {test_dates[-1].date()}")
    
    results = {'dates': [], 'port_ret': [], 'bench_ret': [], 'num_stocks': []}
    
    for test_date in test_dates:
        print(f"\n{'='*60}")
        print(f"Rebalancing: {test_date.date()}")
        
        # Strict PIT training filter
        train_mask = (df['outcome_date'] < test_date) & (df['3mo_return'].notna())
        test_mask = df['public_date'] == test_date
        
        train_df = df[train_mask].copy()
        test_df = df[test_mask].copy()
        
        if len(train_df) < 500:
            print(f"  Skipping: Insufficient training data ({len(train_df)} rows)")
            continue
        
        if len(test_df) < portfolio_size:
            print(f"  Skipping: Insufficient test stocks ({len(test_df)} available)")
            continue
        
        # Initialize model
        rf = StockSelectionRF(n_estimators=150, max_depth=10)
        feats = rf.prepare_features(train_df)
        
        # Check macro availability
        macro_present = [c for c in rf.macro_columns 
                        if c in train_df.columns and train_df[c].notna().any()]
        if macro_present:
            print(f"  Macro features available: {macro_present}")
        else:
            print("  [Warn] No macro data found in training set")
        
        # Prepare data
        X_train = train_df[feats].copy()
        y_train = train_df['3mo_return'].copy()
        meta_train = train_df[['sector', 'public_date']].copy()
        
        # Handle missing data
        X_train_clean, dropped = rf.handle_missing_data(X_train)
        valid_feats = X_train_clean.columns.tolist()
        
        # Align test data
        X_test = test_df[valid_feats].copy()
        X_test_clean = X_test.fillna(rf.feature_medians_).fillna(0)
        
        # Align indices
        y_train = y_train.loc[X_train_clean.index]
        meta_train = meta_train.loc[X_train_clean.index]
        
        y_test = test_df.loc[X_test_clean.index, '3mo_return']
        meta_test = test_df.loc[X_test_clean.index, ['sector', 'public_date']]
        
        # Train
        rf.train(X_train_clean, y_train, meta_train)
        
        # Select portfolio
        top_stocks = rf.get_top_stocks(X_test_clean, X_test_clean.index, 
                                       meta_test, n=portfolio_size)
        
        # Save selections
        save_selections(test_df.loc[top_stocks.index], top_stocks, test_date)
        
        # Calculate returns
        actual_rets = y_test.loc[top_stocks.index]
        port_ret = actual_rets.mean()
        
        # Benchmark (equal-weight all test stocks)
        bench_ret = y_test.mean()
        
        results['dates'].append(test_date)
        results['port_ret'].append(port_ret)
        results['bench_ret'].append(bench_ret)
        results['num_stocks'].append(len(top_stocks))
        
        print(f"  Strategy: {port_ret*100:+.2f}% | Benchmark: {bench_ret*100:+.2f}% | Stocks: {len(top_stocks)}")

    return pd.DataFrame(results)

def save_selections(stock_df, preds, date):
    """Save selected stocks to CSV"""
    os.makedirs('reports_quarterly', exist_ok=True)
    
    cols = ['TICKER', 'sector', 'MthCap', 'pe_inc', 'roe', 'ps', 'roa']
    cols = [c for c in cols if c in stock_df.columns]
    
    out = stock_df[cols].copy()
    out['Predicted_Score'] = preds['score'].values
    out['Date'] = date
    
    filename = f'results/quarterly/selections_{date.date()}.csv'
    out.to_csv(filename, index=False)

# ==============================================================================
# 4. PLOTTING & METRICS
# ==============================================================================
def plot_results(results):
    """Plot cumulative returns"""
    # Remove NaN values
    valid_idx = [i for i, r in enumerate(results['port_ret']) if not np.isnan(r)]
    
    dates_clean = [results['dates'].iloc[i] for i in valid_idx]
    port_rets_clean = [results['port_ret'].iloc[i] for i in valid_idx]
    bench_rets_clean = [results['bench_ret'].iloc[i] for i in valid_idx]
    
    if not port_rets_clean:
        print("No valid returns to plot")
        return
    
    port_cum = np.cumprod(1 + np.array(port_rets_clean))
    bench_cum = np.cumprod(1 + np.array(bench_rets_clean))
    
    plt.figure(figsize=(12, 6))
    plt.plot(dates_clean, port_cum, label='Macro-Aware Strategy', linewidth=2)
    plt.plot(dates_clean, bench_cum, 'k--', label='Equal-Weight Benchmark', linewidth=2)
    plt.title('Quarterly Rebalancing Strategy (TTM + Internal Macro)', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Growth of $1', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/figures/macro_strategy_results.png', dpi=300)
    plt.show()
    
    # Print metrics
    total_ret = (port_cum[-1] - 1) * 100
    bench_total = (bench_cum[-1] - 1) * 100
    ann_ret = ((port_cum[-1] ** (4/len(port_rets_clean))) - 1) * 100
    
    excess = np.array(port_rets_clean) - np.array(bench_rets_clean)
    sharpe = np.mean(excess) / (np.std(excess) + 1e-8) * np.sqrt(4)
    
    print("\n" + "="*60)
    print("PERFORMANCE SUMMARY")
    print("="*60)
    print(f"Strategy Total Return:    {total_ret:+.2f}%")
    print(f"Benchmark Total Return:   {bench_total:+.2f}%")
    print(f"Strategy Ann. Return:     {ann_ret:+.2f}%")
    print(f"Sharpe Ratio:             {sharpe:.2f}")
    print(f"Win Rate vs Benchmark:    {np.mean(excess > 0)*100:.1f}%")
    print(f"Complete Periods:         {len(port_rets_clean)}")
    print("="*60)

# ==============================================================================
# 5. MAIN EXECUTION
# ==============================================================================
if __name__ == "__main__":
    print("="*80)
    print("QUARTERLY MACRO-AWARE STOCK SELECTION")
    print("Using Internal Macro Data from CSV")
    print("="*80 + "\n")
    
    # Load data
    df = load_and_prepare_data('data/simfin_panel.csv')
    
    # Verify macro columns
    macro_cols = ['FEDFUNDS', 'DGS10', '1mo_inf_rate', '1yr_inf_rate', '1mo_GDP', '1yr_GDP']
    present = [c for c in macro_cols if c in df.columns]
    
    print(f"\nMacro Columns Present: {present}")
    if present:
        for col in present:
            non_null = df[col].notna().sum()
            print(f"  {col}: {non_null} non-null values ({non_null/len(df)*100:.1f}%)")
    else:
        print("  [WARNING] No macro columns found - strategy will use fundamentals only")
    
    # Run backtest
    results = quarterly_backtest_internal_macro(
        df, 
        portfolio_size=10,
        start_year=2018,
        min_market_cap=200e6  # $200M minimum
    )
    
    # Plot and report
    plot_results(results)
    
    print("\nBacktest complete. Stock selections saved to 'reports_quarterly/' directory.")
