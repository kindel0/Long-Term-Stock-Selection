import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import calendar
import warnings
import os

warnings.filterwarnings('ignore')

# ==============================================================================
# 1. CORE RANDOM FOREST CLASS
# ==============================================================================
class StockSelectionRF:
    """
    Random Forest model for long-term stock selection based on fundamental analysis.
    Enhanced version with sector neutralization and rank-based targets.
    """
    
    def __init__(self, n_estimators=500, max_depth=12, random_state=42):
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
        self.feature_importances = None
        self.feature_medians_ = None
        self.sector_stats_ = None 
        
    def prepare_features(self, df):
        valuation = ['bm', 'ptb', 'ps', 'pcf', 'pe_inc', 'pe_exi', 'pe_op_basic', 
                     'pe_op_dil', 'evm', 'dpr', 'divyield']
        
        profitability = ['npm', 'gpm', 'roa', 'roe', 'roce', 'opmbd', 'opmad', 
                        'ptpm', 'cfm', 'efftax', 'GProf']
        
        solvency = ['de_ratio', 'debt_at', 'debt_assets', 'debt_capital', 
                   'capital_ratio', 'intcov', 'intcov_ratio', 'dltt_be']
        
        liquidity = ['curr_ratio', 'quick_ratio', 'cash_ratio', 'cash_conversion']
        
        efficiency = ['at_turn', 'inv_turn', 'rect_turn', 'pay_turn', 
                     'sale_invcap', 'sale_equity', 'sale_nwc']
        
        financial_soundness = ['cash_lt', 'invt_act', 'rect_act', 'short_debt', 
                              'curr_debt', 'lt_debt', 'profit_lct', 'ocf_lct', 
                              'cash_debt', 'fcf_ocf']
        
        other = ['accrual', 'rd_sale', 'lt_ppent']
        macro = ['FEDFUNDS', 'DGS10', '1mo_inf_rate', '1yr_inf_rate', '1mo_GDP', '1yr_GDP']
        size = ['MthCap']
        
        all_features = (valuation + profitability + solvency + liquidity + 
                       efficiency + financial_soundness + other + macro + size)
        
        feature_cols = [col for col in all_features if col in df.columns]
        return feature_cols
    
    def neutralize_features(self, X, metadata):
        if 'sector' not in metadata.columns:
            return X
        
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
        lower_bound = np.percentile(y.dropna(), lower_pct)
        upper_bound = np.percentile(y.dropna(), upper_pct)
        return y.clip(lower=lower_bound, upper=upper_bound)
    
    def rank_target(self, y, dates):
        temp_df = pd.DataFrame({'target': y, 'date': dates})
        temp_df['rank'] = temp_df.groupby('date')['target'].rank(pct=True)
        return temp_df['rank']
    
    def handle_missing_data(self, X, missing_threshold_pct=0.5):
        initial_features = X.shape[1]
        missing_pct = X.isnull().sum() / len(X)
        
        features_to_keep = missing_pct[missing_pct <= missing_threshold_pct].index.tolist()
        dropped_features = missing_pct[missing_pct > missing_threshold_pct].index.tolist()
        
        X_filtered = X[features_to_keep].copy()
        
        row_completeness = X_filtered.notna().sum(axis=1) / X_filtered.shape[1]
        rows_to_keep = row_completeness >= 0.8
        X_filtered = X_filtered[rows_to_keep].copy()
        
        if not hasattr(self, 'feature_medians_') or self.feature_medians_ is None:
            self.feature_medians_ = X_filtered.median()
        
        X_clean = X_filtered.fillna(self.feature_medians_)
        if X_clean.isnull().sum().sum() > 0:
            X_clean = X_clean.fillna(0)
            
        return X_clean, dropped_features

    def train(self, X_train, y_train, metadata_train):
        valid_mask = y_train.notna()
        X_train = X_train[valid_mask]
        y_train = y_train[valid_mask]
        metadata_train = metadata_train[valid_mask] if metadata_train is not None else None

        print(f"  Training RF on {len(X_train)} samples with {X_train.shape[1]} features...")
        
        X_train_neutral = self.neutralize_features(X_train, metadata_train)
        y_train_winsorized = self.winsorize_target(y_train)
        y_train_ranked = self.rank_target(y_train_winsorized, metadata_train['public_date'])
        
        self.feature_columns = X_train_neutral.columns.tolist()
        self.model.fit(X_train_neutral, y_train_ranked)
        
        self.feature_importances = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
    def predict(self, X_test, metadata_test=None):
        X_test_neutral = self.neutralize_features(X_test, metadata_test)
        if self.feature_columns:
            for col in self.feature_columns:
                if col not in X_test_neutral.columns:
                    X_test_neutral[col] = 0
            X_test_neutral = X_test_neutral[self.feature_columns]
            
        return self.model.predict(X_test_neutral)
    
    def get_top_stocks(self, X_test, test_indices, metadata_test=None, n=30):
        predictions = self.predict(X_test, metadata_test)
        results = pd.DataFrame({'predicted_rank': predictions}, index=test_indices)
        return results.nlargest(n, 'predicted_rank')

# ==============================================================================
# 2. DATA PREPARATION WITH STRICT PIT + TTM FIXES
# ==============================================================================
def load_and_prepare_data(filepath):
    """
    Loads data AND applies critical TTM (Seasonality) and Target (PIT) fixes.
    """
    print("Loading data...")
    df = pd.read_csv(filepath)
    
    print(f"  Initial shape: {df.shape}")
    print(f"  Available columns: {df.columns.tolist()[:10]}...")  # Show first 10
    
    # 1. Standardize Dates
    print("  > Standardizing dates...")
    df['public_date'] = pd.to_datetime(df['public_date']) + pd.offsets.MonthEnd(0)
    df = df.sort_values(['TICKER', 'public_date'])

    # 2. DETECT FREQUENCY
    print("  > Detecting data frequency...")
    sample_tickers = df['TICKER'].unique()[:10]
    diffs = []
    for ticker in sample_tickers:
        ticker_df = df[df['TICKER'] == ticker]
        if len(ticker_df) > 1:
            diffs.append(ticker_df['public_date'].diff().median().days)
    
    avg_diff = np.median(diffs) if diffs else 30
    is_monthly = avg_diff < 45
    shift_periods = -3 if is_monthly else -1
    print(f"  Detected frequency: {'Monthly' if is_monthly else 'Quarterly'} (avg gap: {avg_diff:.0f} days)")

    # 3. CALCULATE TTM FOR FLOW VARIABLES
    print("  > Calculating TTM (Trailing Twelve Months) Flows...")
    
    # Map potential column names in your data
    flow_candidates = {
        'Revenue': ['Revenue', 'Sales', 'Total Revenue'],
        'Net Income': ['Net Income', 'Net_Income', 'Net Income (Common)'],
        'Gross Profit': ['Gross Profit', 'Gross_Profit'],
        'Operating Income': ['Operating Income', 'Operating_Income', 'EBIT']
    }
    
    ttm_calculated = []
    for standard_name, possible_names in flow_candidates.items():
        raw_col = None
        for name in possible_names:
            if name in df.columns:
                raw_col = name
                break
        
        if raw_col:
            ttm_name = f'{standard_name}_TTM'
            df[ttm_name] = df.groupby('TICKER')[raw_col].transform(
                lambda x: x.rolling(window=4, min_periods=4).sum()
            )
            ttm_calculated.append(ttm_name)
            print(f"    Calculated {ttm_name} from '{raw_col}'")
    
    print(f"  > Successfully calculated {len(ttm_calculated)} TTM variables")

    # 4. RE-CALCULATE KEY RATIOS USING TTM DATA
    print("  > Re-calculating ratios with TTM data...")
    
    def safe_div(a, b): 
        return np.where(b == 0, np.nan, a / b)
    
    # Find equity and asset columns
    equity_col = None
    for col in ['Total Equity', 'Equity', 'Shareholders Equity']:
        if col in df.columns:
            equity_col = col
            break
    
    assets_col = None
    for col in ['Total Assets', 'Assets']:
        if col in df.columns:
            assets_col = col
            break
    
    # P/E Ratio
    if 'Net Income_TTM' in df.columns and 'MthCap' in df.columns:
        pe_ttm = safe_div(df['MthCap'], df['Net Income_TTM'])
        if not pd.Series(pe_ttm).isna().all():
            df['pe_inc'] = pe_ttm
            print("    Updated pe_inc with TTM")
    
    # P/S Ratio
    if 'Revenue_TTM' in df.columns and 'MthCap' in df.columns:
        ps_ttm = safe_div(df['MthCap'], df['Revenue_TTM'])
        if not pd.Series(ps_ttm).isna().all():
            df['ps'] = ps_ttm
            print("    Updated ps with TTM")
        
    # ROE
    if 'Net Income_TTM' in df.columns and equity_col:
        roe_ttm = safe_div(df['Net Income_TTM'], df[equity_col])
        if not pd.Series(roe_ttm).isna().all():
            df['roe'] = roe_ttm
            print("    Updated roe with TTM")
        
    # ROA
    if 'Net Income_TTM' in df.columns and assets_col:
        roa_ttm = safe_div(df['Net Income_TTM'], df[assets_col])
        if not pd.Series(roa_ttm).isna().all():
            df['roa'] = roa_ttm
            print("    Updated roa with TTM")

    # Net Profit Margin
    if 'Net Income_TTM' in df.columns and 'Revenue_TTM' in df.columns:
        npm_ttm = safe_div(df['Net Income_TTM'], df['Revenue_TTM'])
        if not pd.Series(npm_ttm).isna().all():
            df['npm'] = npm_ttm
            print("    Updated npm with TTM")

    # Gross Margin
    if 'Gross Profit_TTM' in df.columns and 'Revenue_TTM' in df.columns:
        gpm_ttm = safe_div(df['Gross Profit_TTM'], df['Revenue_TTM'])
        if not pd.Series(gpm_ttm).isna().all():
            df['gpm'] = gpm_ttm
            print("    Updated gpm with TTM")
    
    # Asset Turnover
    if 'Revenue_TTM' in df.columns and assets_col:
        at_turn_ttm = safe_div(df['Revenue_TTM'], df[assets_col])
        if not pd.Series(at_turn_ttm).isna().all():
            df['at_turn'] = at_turn_ttm
            print("    Updated at_turn with TTM")

    # 5. CALCULATE 3-MONTH FORWARD TARGET WITH VALIDATION
    print("  > Calculating 3-Month Forward Returns...")
    
    price_col = None
    for col in ['MthPrc', 'Adj. Close', 'Adj Close', 'Close']:
        if col in df.columns:
            price_col = col
            break
    
    if not price_col:
        raise ValueError("No price column found in data")
    
    print(f"    Using price column: '{price_col}'")
    
    # Calculate next price and date
    df['next_price'] = df.groupby('TICKER')[price_col].shift(shift_periods)
    df['next_date'] = df.groupby('TICKER')['public_date'].shift(shift_periods)
    
    # Calculate time gap
    df['days_gap'] = (df['next_date'] - df['public_date']).dt.days
    
    # Only keep returns where gap is roughly 3 months (80-105 days)
    valid_gap = (df['days_gap'] >= 80) & (df['days_gap'] <= 105)
    df['3mo_return'] = np.where(
        valid_gap, 
        (df['next_price'] - df[price_col]) / df[price_col], 
        np.nan
    )
    
    valid_returns = df['3mo_return'].notna().sum()
    print(f"    Valid 3-month returns: {valid_returns} ({valid_returns/len(df)*100:.1f}%)")
    
    # 6. STRICT POINT-IN-TIME DATE (WITH REPORTING LAG)
    print("  > Setting outcome dates (with 3-day reporting lag buffer)...")
    df['outcome_date'] = df['public_date'] + pd.DateOffset(months=3) + pd.DateOffset(days=3)
    
    # Clean up temporary columns
    df.drop(['next_price', 'next_date', 'days_gap'], axis=1, inplace=True)

    print(f"  Final shape: {df.shape}")
    return df

def infer_sector_from_data(df):
    if 'sector' in df.columns: 
        return df
    if 'industry' in df.columns:
        df['sector'] = df['industry']
        return df
    
    print("  > Creating pseudo-sectors from size/profitability...")
    df['size_bucket'] = pd.qcut(
        df.groupby('TICKER')['MthCap'].transform('median'), 
        q=3, labels=['Small', 'Mid', 'Large'], duplicates='drop'
    )
    df['profit_bucket'] = pd.qcut(
        df.groupby('TICKER')['roe'].transform('median'), 
        q=3, labels=['Low', 'Med', 'High'], duplicates='drop'
    )
    df['sector'] = df['size_bucket'].astype(str) + '-' + df['profit_bucket'].astype(str)
    return df

# ==============================================================================
# 3. QUARTERLY BACKTEST ENGINE (STRICT PIT)
# ==============================================================================
import yfinance as yf  # <--- Make sure to import this at the top

def quarterly_expanding_window_backtest(df, portfolio_sizes=[25], 
                                      start_year=2018, end_year=2025,
                                      min_market_cap='Mid Cap'):
    
    # 1. Filter Universe
    cap_hierarchy = ['Nano Cap', 'Micro Cap', 'Small Cap', 'Mid Cap', 'Large Cap', 'Mega Cap']
    if 'cap' in df.columns and min_market_cap != 'Nano Cap':
        try:
            allowed = cap_hierarchy[cap_hierarchy.index(min_market_cap):]
            df = df[df['cap'].isin(allowed)].copy()
            print(f"Filtered Universe: {min_market_cap}+ ({len(df)} rows)")
        except ValueError:
            print(f"  Warning: '{min_market_cap}' not found in cap hierarchy, using all data")

    df = infer_sector_from_data(df)

    # 2. FETCH BENCHMARK DATA (S&P 500)
    print("\nDownloading S&P 500 Benchmark Data (^GSPC)...")
    try:
        # Download buffer: Start 1 year early to ensure we have coverage
        sp500_df = yf.download('^GSPC', start=f"{start_year-1}-01-01", end=None, progress=False)
        
        # Handle potential MultiIndex columns in new yfinance versions
        if isinstance(sp500_df.columns, pd.MultiIndex):
            bench_prices = sp500_df['Adj Close'].iloc[:, 0] if 'Adj Close' in sp500_df else sp500_df['Close'].iloc[:, 0]
        else:
            bench_prices = sp500_df['Adj Close'] if 'Adj Close' in sp500_df else sp500_df['Close']
            
        print(f"  > Loaded {len(bench_prices)} days of S&P 500 data.")
        use_sp500 = True
    except Exception as e:
        print(f"  > Warning: Could not download S&P 500 data. Using Universe Average instead. Error: {e}")
        use_sp500 = False
        bench_prices = None

    # Generate Quarterly Test Dates
    all_dates = sorted(df['public_date'].unique())
    test_dates = [d for d in all_dates if d.year >= start_year and d.year <= end_year and d.month in [3, 6, 9, 12]]
    
    print(f"\nBacktest Timeline: {len(test_dates)} Rebalances from {test_dates[0].date()} to {test_dates[-1].date()}")

    results = {size: {'predictions': [], 'actual_returns': [], 'dates': []} for size in portfolio_sizes}
    results['benchmark'] = {'actual_returns': [], 'dates': []}

    for test_date in test_dates:
        print(f"\n{'='*60}")
        print(f"Rebalancing Date: {test_date.date()}")
        
        # Strict PIT Training
        train_mask = (df['outcome_date'] < test_date) & (df['3mo_return'].notna())
        test_mask = df['public_date'] == test_date
        
        train_df = df[train_mask].copy()
        test_df = df[test_mask].copy()
        
        if len(train_df) < 500 or len(test_df) < 50:
            print(f"  Skipping - Insufficient data (Train: {len(train_df)}, Test: {len(test_df)})")
            continue
            
        print(f"  Training on {len(train_df)} historical samples (Strict PIT)")
        
        # Initialize & Train Model
        rf_model = StockSelectionRF(n_estimators=200, max_depth=10)
        feature_cols = rf_model.prepare_features(train_df)
        
        X_train = train_df[feature_cols].copy()
        y_train = train_df['3mo_return'].copy()
        meta_train = train_df[['sector', 'public_date']].copy()
        
        X_test = test_df[feature_cols].copy()
        y_test = test_df['3mo_return'].copy()
        meta_test = test_df[['sector', 'public_date']].copy()
        
        X_train_clean, dropped = rf_model.handle_missing_data(X_train)
        valid_feats = X_train_clean.columns.tolist()
        X_test_clean = X_test[valid_feats].fillna(rf_model.feature_medians_).fillna(0)
        
        y_train = y_train.loc[X_train_clean.index]
        meta_train = meta_train.loc[X_train_clean.index]
        y_test = y_test.loc[X_test_clean.index]
        meta_test = meta_test.loc[X_test_clean.index]
        
        rf_model.train(X_train_clean, y_train, meta_train)
        
        # Test Portfolios
        for size in portfolio_sizes:
            if len(X_test_clean) < size: continue
            
            top_stocks = rf_model.get_top_stocks(X_test_clean, X_test_clean.index, meta_test, n=size)
            actual_rets = y_test.loc[top_stocks.index]
            port_ret = actual_rets.mean()
            
            save_stock_selections_during_backtest(
                X_test_clean, top_stocks, y_test, test_df, 
                test_date.strftime('%Y-%m-%d'), size, 
                'restricted' if min_market_cap != 'Nano Cap' else 'unrestricted'
            )
            
            results[size]['actual_returns'].append(port_ret)
            results[size]['dates'].append(test_date)
            results[size]['predictions'].append(top_stocks['predicted_rank'].values)
            
            print(f"  Portfolio {size}: {port_ret*100:+.2f}% Return")
            
        # --- BENCHMARK CALCULATION (S&P 500) ---
        if use_sp500 and bench_prices is not None:
            # We need the return from test_date -> test_date + 3 months
            entry_date = test_date
            exit_date = test_date + pd.DateOffset(months=3)
            
            # Find nearest available trading days in SP500 data
            try:
                # get_indexer with method='nearest' handles weekends/holidays automatically
                entry_idx = bench_prices.index.get_indexer([entry_date], method='nearest')[0]
                exit_idx = bench_prices.index.get_indexer([exit_date], method='nearest')[0]
                
                # Ensure we aren't looking into the future (beyond downloaded data)
                if exit_date > bench_prices.index[-1]:
                     bench_ret = np.nan
                else:
                    p_entry = bench_prices.iloc[entry_idx]
                    p_exit = bench_prices.iloc[exit_idx]
                    bench_ret = (p_exit - p_entry) / p_entry
            except IndexError:
                bench_ret = np.nan
        else:
            # Fallback to universe average if yfinance failed
            bench_ret = y_test.mean()

        results['benchmark']['actual_returns'].append(bench_ret)
        results['benchmark']['dates'].append(test_date)
        
        bench_label = "S&P 500" if use_sp500 else "Univ Avg"
        print(f"  Benchmark ({bench_label}):    {bench_ret*100:+.2f}% Return")

    return results

# ==============================================================================
# 4. REPORTING UTILS
# ==============================================================================
def save_stock_selections_during_backtest(X_test_clean, top_stocks, y_test_clean, 
                                          test_df, test_date_str, portfolio_size, 
                                          universe_name, output_dir='results/quarterly'):
    os.makedirs(output_dir, exist_ok=True)
    selected_indices = top_stocks.index
    selected_stocks = test_df.loc[selected_indices].copy()
    
    selected_stocks['Predicted_Rank'] = top_stocks['predicted_rank'].values
    selected_stocks['Actual_3mo_Return'] = y_test_clean.loc[selected_indices].values
    
    cols_to_save = ['TICKER', 'public_date', 'sector', 'MthCap', 'pe_inc', 'roe', 'Predicted_Rank', 'Actual_3mo_Return']
    cols_to_save = [c for c in cols_to_save if c in selected_stocks.columns]
    
    report_df = selected_stocks[cols_to_save].copy()
    report_df.insert(0, 'Test_Date', test_date_str)
    
    filename = f'{output_dir}/{universe_name}_date{test_date_str}_p{portfolio_size}.csv'
    report_df.to_csv(filename, index=False)

def plot_results(results, save_path=None):
    plt.figure(figsize=(12, 6))
    
    # Benchmark
    b_dates = results['benchmark']['dates']
    b_rets = results['benchmark']['actual_returns']
    if b_rets:
        # Remove NaN values
        valid_idx = [i for i, r in enumerate(b_rets) if not np.isnan(r)]
        b_dates_clean = [b_dates[i] for i in valid_idx]
        b_rets_clean = [b_rets[i] for i in valid_idx]
        
        if b_rets_clean:
            b_cum = np.cumprod(1 + np.array(b_rets_clean))
            plt.plot(b_dates_clean, b_cum, 'k--', label='Benchmark', linewidth=2)
    
    # Portfolios
    for size in results:
        if size == 'benchmark': continue
        p_dates = results[size]['dates']
        p_rets = results[size]['actual_returns']
        if p_rets:
            # Remove NaN values
            valid_idx = [i for i, r in enumerate(p_rets) if not np.isnan(r)]
            p_dates_clean = [p_dates[i] for i in valid_idx]
            p_rets_clean = [p_rets[i] for i in valid_idx]
            
            if p_rets_clean:
                p_cum = np.cumprod(1 + np.array(p_rets_clean))
                plt.plot(p_dates_clean, p_cum, label=f'Top {size} Stocks', linewidth=2)
                total_ret = (p_cum[-1]-1)*100
                ann_ret = ((p_cum[-1]**(4/len(p_rets_clean)))-1)*100
                print(f"\nTop {size} Performance:")
                print(f"  Total Return: {total_ret:.2f}%")
                print(f"  Annualized Return: {ann_ret:.2f}%")
                print(f"  Complete Periods: {len(p_rets_clean)}")
            
    plt.title('Quarterly Rebalancing Strategy (Strict PIT + TTM Fix)', fontsize=14, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Growth of $1', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path: 
        plt.savefig(save_path, dpi=300)
    plt.show()

def calculate_performance_metrics(results):
    metrics = []
    bench_rets = np.array(results['benchmark']['actual_returns'])
    
    # Remove NaN values (incomplete periods)
    bench_rets = bench_rets[~np.isnan(bench_rets)]
    
    for size in results:
        if size == 'benchmark': continue
        rets = np.array(results[size]['actual_returns'])
        if len(rets) == 0: continue
        
        # Remove NaN values
        rets = rets[~np.isnan(rets)]
        if len(rets) == 0: continue
        
        excess = rets - bench_rets[:len(rets)]
        
        metrics.append({
            'Portfolio': f'Top {size}',
            'Total Return (%)': (np.prod(1+rets)-1)*100,
            'Ann. Return (%)': ((np.prod(1+rets)**(4/len(rets)))-1)*100,
            'Sharpe Ratio': np.mean(excess) / (np.std(excess) + 1e-8) * np.sqrt(4),
            'Win Rate (%)': np.mean(excess > 0)*100,
            'Avg Quarterly Return (%)': np.mean(rets)*100,
            'Num Periods': len(rets)
        })
    
    # Add benchmark
    if len(bench_rets) > 0:
        metrics.append({
            'Portfolio': 'Benchmark',
            'Total Return (%)': (np.prod(1+bench_rets)-1)*100,
            'Ann. Return (%)': ((np.prod(1+bench_rets)**(4/len(bench_rets)))-1)*100,
            'Sharpe Ratio': np.nan,
            'Win Rate (%)': np.nan,
            'Avg Quarterly Return (%)': np.mean(bench_rets)*100,
            'Num Periods': len(bench_rets)
        })
    
    return pd.DataFrame(metrics)

# ==============================================================================
# 5. MAIN EXECUTION
# ==============================================================================
if __name__ == "__main__":
    print("="*80)
    print("QUARTERLY STOCK SELECTION BACKTEST")
    print("Strict Point-in-Time + TTM Seasonality Adjustment")
    print("="*80 + "\n")
    
    # 1. Load Data with Fixes
    df = load_and_prepare_data('data/simfin_panel.csv')
    
    # 2. Run Backtest
    results = quarterly_expanding_window_backtest(
        df, 
        portfolio_sizes=[10],
        start_year=2018,
        end_year=2025,
        min_market_cap='Mid Cap'
    )
    
    # 3. Display Metrics
    print("\n" + "="*80)
    print("PERFORMANCE METRICS")
    print("="*80)
    metrics = calculate_performance_metrics(results)
    print(metrics.to_string(index=False))
    
    # 4. Plot
    print("\n" + "="*80)
    print("GENERATING CHART")
    print("="*80)
    plot_results(results, save_path='results/figures/quarterly_results_fixed.png')
    
    print("\nBacktest complete. Results saved to 'reports_quarterly/' directory.")
