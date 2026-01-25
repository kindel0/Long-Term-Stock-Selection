import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class StockSelectionRF:
    """
    Random Forest model for long-term stock selection based on fundamental analysis.
    Enhanced version with sector neutralization and rank-based targets to prevent
    market timing bias and improve stock picking performance.
    """
    
    def __init__(self, n_estimators=500, max_depth=12, random_state=42):
        """
        Initialize Random Forest model with robust hyperparameters.
        
        Parameters:
        -----------
        n_estimators : int
            Number of trees (increased from 350 to 500)
        max_depth : int
            Maximum depth (reduced from 13 to 12 for stability)
        random_state : int
            Random seed for reproducibility
        """
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,      # 500 trees for stability
            max_depth=max_depth,             # 12 prevents overfitting to macro regimes
            max_features=0.3,                # Force feature diversity (30% per split)
            min_samples_leaf=50,             # Ensure broad, stable rules
            max_samples=0.7,                 # Bootstrap 70% of data
            random_state=random_state,
            n_jobs=-1
        )
        self.feature_columns = None
        self.feature_importances = None
        self.feature_medians_ = None
        self.sector_stats_ = None  # Store sector neutralization statistics
        
        # CHANGE 1: Add macro columns tracking
        self.macro_columns = ['FEDFUNDS', 'DGS10', '1mo_inf_rate', '1yr_inf_rate', '1mo_GDP', '1yr_GDP']
        
    def prepare_features(self, df):
        """
        Select and prepare features for modeling.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe with all columns
            
        Returns:
        --------
        feature_cols : list
            List of feature column names
        """
        # Financial ratios (from paper's Appendix A)
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
        
        # Macroeconomic indicators - REDUCED weight via max_features
        macro = ['FEDFUNDS', 'DGS10', '1mo_inf_rate', '1yr_inf_rate', 
                '1mo_GDP', '1yr_GDP']
        
        # Market cap
        size = ['MthCap']
        
        # Combine all feature groups
        all_features = (valuation + profitability + solvency + liquidity + 
                       efficiency + financial_soundness + other + macro + size)
        
        # Only keep features that exist in the dataframe
        feature_cols = [col for col in all_features if col in df.columns]
        
        return feature_cols
    
    def neutralize_features(self, X, metadata):
        """
        Apply sector neutralization to features.
        
        For each feature, calculate Z-score within each (sector, date) group.
        This ensures the model compares stocks to their sector peers rather than
        the entire market, preventing sector rotation bias.
        
        IMPORTANT: Macro features are NOT neutralized - they represent global
        economic context and should be passed through as raw values.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Feature dataframe
        metadata : pd.DataFrame
            Dataframe with 'sector' and 'public_date' columns, same index as X
            
        Returns:
        --------
        X_neutralized : pd.DataFrame
            Sector-neutralized features (with macro features kept raw)
        """
        if 'sector' not in metadata.columns:
            print("Warning: No sector column found. Skipping sector neutralization.")
            return X
        
        X_neutral = X.copy()
        
        # Merge metadata with features for grouping
        combined = X.copy()
        combined['sector'] = metadata['sector']
        combined['public_date'] = metadata['public_date']
        
        # For each feature, calculate Z-score within (sector, date) groups
        for col in X.columns:
            if col in ['sector', 'public_date']:
                continue
            
            # CHANGE 2: Skip macro features - they are global context
            # Macro data is the same for all stocks on a date
            # Z-scoring it would destroy its predictive value
            if col in self.macro_columns:
                continue
                
            # Group by sector and date
            grouped = combined.groupby(['sector', 'public_date'])[col]
            
            # Calculate mean and std within each group
            group_mean = grouped.transform('mean')
            group_std = grouped.transform('std')
            
            # Apply Z-score transformation
            # Z = (X - mean) / std, handle division by zero
            X_neutral[col] = (combined[col] - group_mean) / (group_std + 1e-8)
            
            # Fill any remaining NaN/inf values with 0
            X_neutral[col] = X_neutral[col].replace([np.inf, -np.inf], np.nan).fillna(0)
        
        print(f"Applied sector neutralization across {metadata['sector'].nunique()} sectors")
        print(f"Macro features passed through raw (not neutralized): {[c for c in X.columns if c in self.macro_columns]}")
        
        return X_neutral
    
    def winsorize_target(self, y, lower_pct=1, upper_pct=99):
        """
        Winsorize target variable to handle extreme outliers.
        
        Parameters:
        -----------
        y : pd.Series
            Target variable (returns)
        lower_pct : float
            Lower percentile for clipping (default 1st percentile)
        upper_pct : float
            Upper percentile for clipping (default 99th percentile)
            
        Returns:
        --------
        y_winsorized : pd.Series
            Winsorized target variable
        """
        lower_bound = np.percentile(y.dropna(), lower_pct)
        upper_bound = np.percentile(y.dropna(), upper_pct)
        
        y_winsorized = y.clip(lower=lower_bound, upper=upper_bound)
        
        print(f"Winsorized target: [{lower_bound:.4f}, {upper_bound:.4f}]")
        
        return y_winsorized
    
    def rank_target(self, y, dates):
        """
        Convert target to cross-sectional percentile rank within each date.
        
        This forces the model to learn "stock picking" (relative performance)
        rather than "market timing" (absolute returns).
        
        Parameters:
        -----------
        y : pd.Series
            Target variable (returns)
        dates : pd.Series
            Dates for grouping, same index as y
            
        Returns:
        --------
        y_ranked : pd.Series
            Ranked target (0.0 to 1.0 scale)
        """
        # Create temporary dataframe for ranking
        temp_df = pd.DataFrame({
            'target': y,
            'date': dates
        })
        
        # Rank within each date group (0 to 1 scale)
        temp_df['rank'] = temp_df.groupby('date')['target'].rank(pct=True)
        
        y_ranked = temp_df['rank']
        
        print(f"Converted to cross-sectional ranks (mean: {y_ranked.mean():.3f})")
        
        return y_ranked
    
    def handle_missing_data(self, X, missing_threshold_pct=0.5):
        """
        Handle missing data with a more practical approach:
        1. Drop features with > threshold% missing values
        2. For remaining features, impute missing values with median
        
        Parameters:
        -----------
        X : pd.DataFrame
            Feature dataframe
        missing_threshold_pct : float
            Maximum percentage of missing values allowed per feature (0-1)
            
        Returns:
        --------
        X_clean : pd.DataFrame
            Cleaned feature dataframe
        dropped_features : list
            List of dropped feature names
        """
        initial_features = X.shape[1]
        initial_rows = len(X)
        
        # Calculate missing percentage per feature
        missing_pct = X.isnull().sum() / len(X)
        
        # Drop features exceeding threshold
        features_to_keep = missing_pct[missing_pct <= missing_threshold_pct].index.tolist()
        dropped_features = missing_pct[missing_pct > missing_threshold_pct].index.tolist()
        
        X_filtered = X[features_to_keep].copy()
        
        print(f"Dropped {len(dropped_features)} features with >{missing_threshold_pct*100:.0f}% missing values")
        
        # Keep rows that have at least 80% of features non-missing
        row_completeness = X_filtered.notna().sum(axis=1) / X_filtered.shape[1]
        rows_to_keep = row_completeness >= 0.8
        
        X_filtered = X_filtered[rows_to_keep].copy()
        dropped_rows = initial_rows - len(X_filtered)
        
        print(f"Dropped {dropped_rows} instances ({100*dropped_rows/initial_rows:.1f}%) with <80% feature completeness")
        
        # Impute remaining missing values with median
        # Store medians for consistent imputation in test set
        if not hasattr(self, 'feature_medians_') or self.feature_medians_ is None:
            self.feature_medians_ = X_filtered.median()
        
        X_clean = X_filtered.fillna(self.feature_medians_)
        
        remaining_missing = X_clean.isnull().sum().sum()
        if remaining_missing > 0:
            print(f"Warning: {remaining_missing} missing values remain after imputation")
            # Fill any remaining with 0 as last resort
            X_clean = X_clean.fillna(0)
        
        print(f"Final dataset: {len(X_clean)} samples, {X_clean.shape[1]} features")
        
        return X_clean, dropped_features

    def train(self, X_train, y_train, metadata_train):
        """
        Train the Random Forest model with sector neutralization and rank-based targets.
        Excludes rows with missing targets (recent data) to prevent training errors.
        """
        # Filter out rows where y_train is NaN (recent data for prediction)
        valid_mask = y_train.notna()
        
        if valid_mask.sum() < len(y_train):
            skipped = len(y_train) - valid_mask.sum()
            print(f"Skipping {skipped} rows with missing targets (likely recent data for prediction).")
        
        X_train = X_train[valid_mask]
        y_train = y_train[valid_mask]
        metadata_train = metadata_train[valid_mask] if metadata_train is not None else None

        print(f"Training Random Forest on {len(X_train)} samples with {X_train.shape[1]} features...")
        
        # Step 1: Apply sector neutralization
        if metadata_train is not None and 'sector' in metadata_train.columns:
            X_train_neutral = self.neutralize_features(X_train, metadata_train)
        else:
            X_train_neutral = X_train
            print("Warning: No sector data available, skipping neutralization")
        
        # Step 2: Winsorize target
        y_train_winsorized = self.winsorize_target(y_train)
        
        # Step 3: Convert to cross-sectional ranks
        if metadata_train is not None and 'public_date' in metadata_train.columns:
            y_train_ranked = self.rank_target(y_train_winsorized, metadata_train['public_date'])
        else:
            y_train_ranked = y_train_winsorized
            print("Warning: No date data available, skipping ranking")
        
        # Store feature columns
        self.feature_columns = X_train_neutral.columns.tolist()
        
        # Train model on neutralized features with ranked targets
        self.model.fit(X_train_neutral, y_train_ranked)
        
        # Store feature importances
        self.feature_importances = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # CHANGE 3: Report macro feature importance
        macro_features = [f for f in self.feature_columns if f in self.macro_columns]
        if macro_features:
            print(f"\nMacro features in model: {len(macro_features)}")
            for feat in macro_features:
                imp = self.feature_importances[self.feature_importances['feature'] == feat]['importance'].values
                if len(imp) > 0:
                    rank = list(self.feature_importances['feature']).index(feat) + 1
                    print(f"  {feat}: Rank #{rank}, Importance: {imp[0]:.4f}")
        else:
            print("\nNo macro features found in training data")
        
        # Calculate training MSE (on ranked targets)
        train_pred = self.model.predict(X_train_neutral)
        train_mse = mean_squared_error(y_train_ranked, train_pred)
        print(f"Training MSE (on ranks): {train_mse:.4f}")

        
    def predict(self, X_test, metadata_test=None):
        """
        Make predictions on test set with sector neutralization.
        
        Parameters:
        -----------
        X_test : pd.DataFrame
            Test features
        metadata_test : pd.DataFrame, optional
            Metadata with 'sector' and 'public_date' columns
            
        Returns:
        --------
        predictions : np.array
            Predicted ranks (0-1 scale)
        """
        # Apply sector neutralization if metadata available
        if metadata_test is not None and 'sector' in metadata_test.columns:
            X_test_neutral = self.neutralize_features(X_test, metadata_test)
        else:
            X_test_neutral = X_test
        
        return self.model.predict(X_test_neutral)
    
    def get_top_stocks(self, X_test, test_indices, metadata_test=None, n=30):
        """
        Get top n stocks based on predicted ranks.
        
        Parameters:
        -----------
        X_test : pd.DataFrame
            Test features
        test_indices : pd.Index
            Original indices for test set
        metadata_test : pd.DataFrame, optional
            Metadata for sector neutralization
        n : int
            Number of stocks to select
            
        Returns:
        --------
        top_stocks : pd.DataFrame
            Dataframe with top n stock predictions
        """
        predictions = self.predict(X_test, metadata_test)
        
        results = pd.DataFrame({
            'predicted_rank': predictions
        }, index=test_indices)
        
        # Sort by predicted rank and get top n
        top_stocks = results.nlargest(n, 'predicted_rank')
        
        return top_stocks


def load_and_prepare_data(filepath):
    """
    Load and prepare the dataset.
    
    Parameters:
    -----------
    filepath : str
        Path to CSV file
        
    Returns:
    --------
    df : pd.DataFrame
        Prepared dataframe
    """
    print("Loading data...")
    df = pd.read_csv(filepath)
    
    # Convert dates using proper month-end snapping
    df['public_date'] = pd.to_datetime(df['public_date'])
    df['public_date'] = df['public_date'] + pd.offsets.MonthEnd(0)
    
    # Sort by ticker and date
    df = df.sort_values(['TICKER', 'public_date'])
    
    return df


def infer_sector_from_data(df):
    """
    Infer sector classification if not explicitly available.
    Uses industry patterns or creates pseudo-sectors based on company characteristics.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with stock data
        
    Returns:
    --------
    df : pd.DataFrame
        Dataframe with 'sector' column added
    """
    if 'sector' in df.columns:
        print(f"Found existing sector column with {df['sector'].nunique()} sectors")
        return df
    
    # Try to infer from industry or other columns
    if 'industry' in df.columns:
        df['sector'] = df['industry']
        print(f"Using 'industry' column as sector proxy ({df['sector'].nunique()} sectors)")
        return df
    
    # Create pseudo-sectors based on market cap and profitability characteristics
    print("Creating pseudo-sectors based on company characteristics...")
    
    # Simple clustering approach: combine size and profitability
    df['size_bucket'] = pd.qcut(df.groupby('TICKER')['MthCap'].transform('median'), 
                                 q=3, labels=['Small', 'Mid', 'Large'], duplicates='drop')
    
    df['profit_bucket'] = pd.qcut(df.groupby('TICKER')['roe'].transform('median'), 
                                   q=3, labels=['Low', 'Med', 'High'], duplicates='drop')
    
    df['sector'] = df['size_bucket'].astype(str) + '-' + df['profit_bucket'].astype(str)
    df['sector'] = df['sector'].fillna('Unknown')
    
    print(f"Created {df['sector'].nunique()} pseudo-sectors")
    
    return df


def expanding_window_backtest(df, portfolio_sizes=[10, 25, 50, 100, 200], 
                              start_year=2017, end_year=2025,
                              min_market_cap='Nano Cap',
                              reference_month=12,
                              reference_day=31,
                              training_lag_months=3):
    """
    Perform expanding window backtest with sector neutralization and rank-based targets.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Full dataset
    portfolio_sizes : list
        List of portfolio sizes to test
    start_year : int
        First year for testing (needs 2+ years of training data)
    end_year : int
        Last year for testing (will be adjusted based on available data)
    min_market_cap : str
        Minimum market cap category to include
    reference_month : int
        Month to use as reference (1-12, default 12 for December)
    reference_day : int
        Day to use as reference (1-31, default 31)
    training_lag_months : int
        Number of months before test date to use as training cutoff (default 3)
        This accounts for reporting delays and ensures point-in-time integrity
        
    Returns:
    --------
    results : dict
        Dictionary containing results for each portfolio size
    """
    import calendar
    from dateutil.relativedelta import relativedelta
    
    # Define market cap hierarchy
    cap_hierarchy = ['Nano Cap', 'Micro Cap', 'Small Cap', 'Mid Cap', 'Large Cap', 'Mega Cap']
    
    # Infer or create sector information
    df = infer_sector_from_data(df)

    # Normalize cap labels
    if 'cap' in df.columns:
        df['cap'] = df['cap'].astype(str).str.strip()
        df.loc[df['cap'].isin(['nan', 'None', 'NaN']), 'cap'] = np.nan   
    
    # Filter by market cap if column exists
    if 'cap' in df.columns and min_market_cap != 'Nano Cap':

        # Drop rows without cap classification
        before = len(df)
        df = df[df['cap'].notna()].copy()
        dropped = before - len(df)

        if dropped > 0:
            print(f"Dropped {dropped} rows with missing cap for restricted universe")

        min_cap_idx = cap_hierarchy.index(min_market_cap)
        allowed_caps = cap_hierarchy[min_cap_idx:]

        df = df[df['cap'].isin(allowed_caps)].copy()

        print(f"\nFiltering to {min_market_cap} and above:")
        print(df['cap'].value_counts().sort_index())

        # SAFETY ASSERTION
        assert df['cap'].isin(allowed_caps).all(), \
            "Universe filter violated: df contains caps below min_market_cap"
    
    # Find available dates for the reference month
    df['year'] = df['public_date'].dt.year
    df['month'] = df['public_date'].dt.month
    
    # Look for reference month dates
    year_ref_dates = df[df['month'] == reference_month].groupby('year')['public_date'].max()
    available_years = sorted(year_ref_dates.index)
    
    # Filter to requested range
    test_years = [y for y in range(start_year, end_year + 1) if y in available_years]
    
    # Ensure each year has 1yr_return data
    valid_test_years = []
    for year in test_years:
        test_date = year_ref_dates[year]
        has_returns = df[(df['public_date'] == test_date) & (df['1yr_return'].notna())]
        if len(has_returns) > 0:
            valid_test_years.append(year)
    
    print(f"\nAvailable test years with 1yr_return data: {valid_test_years}")
    if len(valid_test_years) == 0:
        print("ERROR: No valid test years found!")
        return {}, None
    
    # Initialize results storage
    results = {size: {'predictions': [], 'actual_returns': [], 'dates': []} 
               for size in portfolio_sizes}
    benchmark_returns = []
    
    # Perform expanding window backtest
    for test_year in valid_test_years:
        print(f"\n{'='*60}")
        print(f"Testing year: {test_year}")
        print(f"{'='*60}")
        
        # Initialize fresh model for each year
        rf_model = StockSelectionRF()
        feature_cols = rf_model.prepare_features(df)
        
        # Define train/test split
        last_day_of_train_month = calendar.monthrange(test_year - 1, reference_month)[1]
        train_end = datetime(test_year - 1, reference_month, last_day_of_train_month)
        test_date = year_ref_dates[test_year]
        
        print(f"Train end date: {train_end.strftime('%Y-%m-%d')}")
        print(f"Test date: {test_date.strftime('%Y-%m-%d')}")
        
        # Split data
        train_mask = df['public_date'] <= train_end
        test_mask = df['public_date'] == test_date
        
        train_df = df[train_mask].copy()
        test_df = df[test_mask].copy()
        
        print(f"Training samples: {len(train_df)}")
        print(f"Test samples: {len(test_df)}")
        
        if len(train_df) < 100 or len(test_df) < max(portfolio_sizes):
            print(f"Skipping {test_year} - insufficient data")
            continue
        
        # Prepare features, target, and metadata
        X_train = train_df[feature_cols].copy()
        y_train = train_df['1yr_return'].copy()
        metadata_train = train_df[['sector', 'public_date']].copy()
        
        X_test = test_df[feature_cols].copy()
        y_test = test_df['1yr_return'].copy()
        metadata_test = test_df[['sector', 'public_date']].copy()
        
        # Handle missing data in training set
        X_train_clean, dropped_features = rf_model.handle_missing_data(X_train)
        
        # Keep only features that weren't dropped
        features_to_use = [f for f in feature_cols if f not in dropped_features]
        
        # Apply same cleaning to test set
        X_test_filtered = X_test[features_to_use].copy()
        
        # Keep test rows with at least 80% completeness
        row_completeness = X_test_filtered.notna().sum(axis=1) / X_test_filtered.shape[1]
        rows_to_keep = row_completeness >= 0.8
        X_test_filtered = X_test_filtered[rows_to_keep]
        
        # Impute test set using training medians
        X_test_clean = X_test_filtered.fillna(rf_model.feature_medians_).fillna(0)
        
        # Align metadata and targets with cleaned features
        y_train_clean = y_train.loc[X_train_clean.index]
        metadata_train_clean = metadata_train.loc[X_train_clean.index]
        
        y_test_clean = y_test.loc[X_test_clean.index]
        metadata_test_clean = metadata_test.loc[X_test_clean.index]

        assert X_test_clean.index.is_unique
        assert X_test_clean.index.equals(metadata_test_clean.index)
        
        print(f"After cleaning - Train: {len(X_train_clean)}, Test: {len(X_test_clean)}")
        
        if len(X_train_clean) < 100 or len(X_test_clean) < 10:
            print(f"Skipping {test_year} - insufficient data after cleaning")
            continue
        
        # Train model with sector neutralization and rank-based targets
        rf_model.train(X_train_clean, y_train_clean, metadata_train_clean)
        
        # Test each portfolio size
        for n_stocks in portfolio_sizes:
            if len(X_test_clean) < n_stocks:
                print(f"Skipping portfolio size {n_stocks} - insufficient test samples")
                continue
            
            # Get top n stocks (predictions are ranks, not returns)
            top_stocks = rf_model.get_top_stocks(
                X_test_clean, 
                X_test_clean.index, 
                metadata_test_clean,
                n=n_stocks
            )
            
            # Get ACTUAL returns for selected stocks
            actual_returns = y_test_clean.loc[top_stocks.index]
            
            # Calculate portfolio return (equal-weighted on ACTUAL returns)
            portfolio_return = actual_returns.mean()

            assert top_stocks.index.isin(X_test_clean.index).all()

            # Save stock selections to report
            universe_label = 'unrestricted' if min_market_cap == 'Nano Cap' else 'restricted'
            save_stock_selections_during_backtest(
                X_test_clean, 
                top_stocks, 
                y_test_clean,
                test_df.loc[X_test_clean.index],
                test_year,
                n_stocks,
                universe_label
            )
            
            # Store results
            results[n_stocks]['predictions'].append(top_stocks['predicted_rank'].values)
            results[n_stocks]['actual_returns'].append(portfolio_return)
            results[n_stocks]['dates'].append(test_year)
            
            print(f"Portfolio size {n_stocks}: {portfolio_return:.2%} return")
        
        # Calculate benchmark return (market average)
        benchmark_return = y_test_clean.mean()
        benchmark_returns.append(benchmark_return)
        print(f"Benchmark return: {benchmark_return:.2%}")
    
    # Add benchmark to results
    results['benchmark'] = {
        'actual_returns': benchmark_returns,
        'dates': valid_test_years[:len(benchmark_returns)]
    }
    
    # Print feature importances from last model
    if hasattr(rf_model, 'feature_importances') and rf_model.feature_importances is not None:
        print("\n" + "="*60)
        print("Top 20 Feature Importances (from final model):")
        print("="*60)
        print(rf_model.feature_importances.head(20).to_string(index=False))
    
    return results, rf_model


def calculate_performance_metrics(results, initial_capital=100):
    """
    Calculate performance metrics following paper's methodology.
    
    Parameters:
    -----------
    results : dict
        Results from backtest
    initial_capital : float
        Initial investment amount
        
    Returns:
    --------
    metrics : pd.DataFrame
        Performance metrics for each portfolio
    """
    metrics_list = []
    
    benchmark_returns = np.array(results['benchmark']['actual_returns'])
    
    for portfolio_size in results.keys():
        if portfolio_size == 'benchmark':
            continue
            
        returns = np.array(results[portfolio_size]['actual_returns'])
        
        if len(returns) == 0:
            continue
        
        # Calculate cumulative returns
        cumulative = initial_capital * np.prod(1 + returns)
        
        # Calculate excess returns
        excess_returns = returns - benchmark_returns[:len(returns)]
        
        # Calculate metrics
        mean_return = returns.mean()
        std_return = returns.std()
        mean_excess = excess_returns.mean()
        
        # Downside deviation (for Sortino-modified IR)
        downside_returns = excess_returns[excess_returns < 0]
        downside_std = np.sqrt(np.mean(downside_returns**2)) if len(downside_returns) > 0 else np.nan
        
        # Sortino-modified Information Ratio
        sir = mean_excess / downside_std if downside_std > 0 else np.nan
        
        # Information Ratio
        ir = mean_excess / std_return if std_return > 0 else np.nan
        
        metrics_list.append({
            'Portfolio Size': portfolio_size,
            'Mean Return': mean_return,
            'Std Return': std_return,
            'Mean Excess Return': mean_excess,
            'Downside Std': downside_std,
            'SIR': sir,
            'IR': ir,
            'Cumulative Value': cumulative,
            'Total Return': (cumulative - initial_capital) / initial_capital
        })
    
    # Add benchmark
    benchmark_cumulative = initial_capital * np.prod(1 + benchmark_returns)
    metrics_list.append({
        'Portfolio Size': 'Benchmark',
        'Mean Return': benchmark_returns.mean(),
        'Std Return': benchmark_returns.std(),
        'Mean Excess Return': 0,
        'Downside Std': np.nan,
        'SIR': np.nan,
        'IR': np.nan,
        'Cumulative Value': benchmark_cumulative,
        'Total Return': (benchmark_cumulative - initial_capital) / initial_capital
    })
    
    metrics_df = pd.DataFrame(metrics_list)
    return metrics_df


def plot_results(results, save_path=None):
    """
    Create visualizations of backtest results.
    
    Parameters:
    -----------
    results : dict
        Results from backtest
    save_path : str, optional
        Path to save plots
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Cumulative Returns
    ax1 = axes[0, 0]
    initial_capital = 100
    
    benchmark_returns = np.array(results['benchmark']['actual_returns'])
    benchmark_dates = results['benchmark']['dates']
    benchmark_cumulative = initial_capital * np.cumprod(1 + benchmark_returns)
    
    ax1.plot(benchmark_dates, benchmark_cumulative, 'k--', label='Benchmark', linewidth=2)
    
    for size in [10, 25, 50, 100, 200]:
        if size not in results or len(results[size]['actual_returns']) == 0:
            continue
        returns = np.array(results[size]['actual_returns'])
        dates = results[size]['dates']
        cumulative = initial_capital * np.cumprod(1 + returns)
        ax1.plot(dates, cumulative, label=f'{size} stocks', linewidth=2)
    
    ax1.set_title('Cumulative Returns', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Annual Returns
    ax2 = axes[0, 1]
    
    width = 0.15
    x = np.arange(len(benchmark_dates))
    
    ax2.bar(x - 2*width, benchmark_returns * 100, width, label='Benchmark', alpha=0.7)
    
    for i, size in enumerate([10, 25, 50, 100, 200]):
        if size not in results or len(results[size]['actual_returns']) == 0:
            continue
        returns = np.array(results[size]['actual_returns']) * 100
        ax2.bar(x + (i-1)*width, returns, width, label=f'{size} stocks', alpha=0.7)
    
    ax2.set_title('Annual Returns', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Return (%)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(benchmark_dates, rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # Plot 3: Excess Returns
    ax3 = axes[1, 0]
    
    for size in [10, 25, 50, 100, 200]:
        if size not in results or len(results[size]['actual_returns']) == 0:
            continue
        returns = np.array(results[size]['actual_returns'])
        excess = (returns - benchmark_returns[:len(returns)]) * 100
        dates = results[size]['dates']
        ax3.plot(dates, excess, marker='o', label=f'{size} stocks', linewidth=2)
    
    ax3.set_title('Excess Returns vs Benchmark', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Year')
    ax3.set_ylabel('Excess Return (%)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='black', linestyle='--', linewidth=1)
    
    # Plot 4: Performance Metrics
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Calculate metrics for display
    metrics_text = "Performance Metrics Summary\n" + "="*40 + "\n\n"
    
    for size in [10, 25, 50, 100, 200]:
        if size not in results or len(results[size]['actual_returns']) == 0:
            continue
        returns = np.array(results[size]['actual_returns'])
        excess = returns - benchmark_returns[:len(returns)]
        
        mean_return = returns.mean() * 100
        mean_excess = excess.mean() * 100
        downside = excess[excess < 0]
        downside_std = np.sqrt(np.mean(downside**2)) * 100 if len(downside) > 0 else 0
        sir = mean_excess / downside_std if downside_std > 0 else 0
        
        metrics_text += f"{size} stocks:\n"
        metrics_text += f"  Mean Return: {mean_return:.2f}%\n"
        metrics_text += f"  Mean Excess: {mean_excess:.2f}%\n"
        metrics_text += f"  SIR: {sir:.3f}\n\n"
    
    benchmark_mean = benchmark_returns.mean() * 100
    metrics_text += f"Benchmark:\n"
    metrics_text += f"  Mean Return: {benchmark_mean:.2f}%\n"
    
    ax4.text(0.1, 0.9, metrics_text, transform=ax4.transAxes, 
             fontsize=11, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def generate_stock_selection_report(df, results, model, universe_name, output_dir='results/annual'):
    """Generate detailed reports of stock selections for each test year."""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    test_years = results['benchmark']['dates']
    all_selections = []
    
    for portfolio_size in [10, 25, 50, 100, 200]:
        if portfolio_size not in results or len(results[portfolio_size]['dates']) == 0:
            continue
        
        print(f"\nGenerating report for {portfolio_size}-stock portfolio...")
        
        years = results[portfolio_size]['dates']
        portfolio_report = []
        
        for idx, year in enumerate(years):
            filename = f'{output_dir}/{universe_name}_year{year}_portfolio{portfolio_size}.csv'

            if not os.path.exists(filename):
                print(f"Missing backtest file: {filename}")
                continue

            selected_df = pd.read_csv(filename)
            portfolio_report.extend(selected_df.to_dict('records'))
            all_selections.extend(selected_df.to_dict('records'))

        portfolio_df = pd.DataFrame(portfolio_report)
        filename = f'{output_dir}/{universe_name}_portfolio_{portfolio_size}_detailed.csv'
        portfolio_df.to_csv(filename, index=False)
        print(f"  Saved: {filename}")
    
    summary_df = pd.DataFrame(all_selections)
    summary_filename = f'{output_dir}/{universe_name}_all_selections_summary.csv'
    summary_df.to_csv(summary_filename, index=False)
    print(f"\nSaved master summary: {summary_filename}")
    
    # Create year-by-year summary statistics
    yearly_summary = []
    for year in test_years:
        year_selections = summary_df[summary_df['Test_Year'] == year]
        
        if len(year_selections) == 0:
            continue
        
        summary_stats = {
            'Test_Year': year,
            'Total_Stocks_Selected': len(year_selections),
            'Unique_Tickers': year_selections['TICKER'].nunique(),
            'Avg_Market_Cap': year_selections['MthCap'].mean(),
            'Median_Market_Cap': year_selections['MthCap'].median(),
            'Avg_BM_Ratio': year_selections['bm'].mean() if 'bm' in year_selections.columns else np.nan,
            'Avg_ROE': year_selections['roe'].mean() if 'roe' in year_selections.columns else np.nan,
            'Avg_Actual_Return': year_selections['Actual_1yr_Return'].mean(),
            'Sectors_Represented': year_selections['sector'].nunique()
        }
        yearly_summary.append(summary_stats)
    
    yearly_summary_df = pd.DataFrame(yearly_summary)
    yearly_filename = f'{output_dir}/{universe_name}_yearly_summary.csv'
    yearly_summary_df.to_csv(yearly_filename, index=False)
    print(f"Saved yearly summary: {yearly_filename}")
    
    return summary_df

def save_stock_selections_during_backtest(X_test_clean, top_stocks, y_test_clean, 
                                          test_df, test_year, portfolio_size, 
                                          universe_name, output_dir='results/annual'):
    """Save stock selections during backtest (call this INSIDE the backtest loop)."""
    import os
    os.makedirs(output_dir, exist_ok=True)

    assert set(top_stocks.index).issubset(set(test_df.index)), \
        "Universe leak detected: selected stocks not in test universe"
    
    selected_indices = top_stocks.index
    selected_stocks = test_df.loc[selected_indices].copy()
    
    selected_stocks['Predicted_Rank'] = top_stocks['predicted_rank'].values
    selected_stocks['Actual_1yr_Return'] = y_test_clean.loc[selected_indices].values
    
    report_cols = [
        'TICKER', 'public_date', 'MthCap', 'cap', 'sector',
        'MthPrc', 'bm', 'ptb', 'ps', 'pcf', 
        'roe', 'roa', 'npm', 'pe_inc',
        'de_ratio', 'curr_ratio', 'quick_ratio',
        'Predicted_Rank', 'Actual_1yr_Return'
    ]
    
    report_cols = [col for col in report_cols if col in selected_stocks.columns]
    report_df = selected_stocks[report_cols].copy()
    
    report_df.insert(0, 'Test_Year', test_year)
    report_df.insert(1, 'Portfolio_Size', portfolio_size)
    
    filename = f'{output_dir}/{universe_name}_year{test_year}_portfolio{portfolio_size}.csv'
    report_df.to_csv(filename, index=False)
    
    return report_df

# CHANGE 4: Main execution with macro verification
if __name__ == "__main__":
    # Load data
    df = load_and_prepare_data('data/simfin_panel.csv')
    
    print(f"\nDataset shape: {df.shape}")
    print(f"Date range: {df['public_date'].min()} to {df['public_date'].max()}")
    print(f"Unique stocks: {df['TICKER'].nunique()}")
    
    # Show sample of missing data
    print(f"\nMissing data overview:")
    missing_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
    print(f"Features with >50% missing: {(missing_pct > 50).sum()}")
    print(f"Features with >25% missing: {(missing_pct > 25).sum()}")
    print(f"\nTop 10 features by missing %:")
    print(missing_pct.head(10))
    
    # CHANGE 4: Verify macro features
    print("\n" + "="*80)
    print("VERIFYING MACRO FEATURES")
    print("="*80)
    macro_cols = ['FEDFUNDS', 'DGS10', '1mo_inf_rate', '1yr_inf_rate', '1mo_GDP', '1yr_GDP']
    macro_present = [c for c in macro_cols if c in df.columns]
    
    if macro_present:
        print(f"Found {len(macro_present)} macro features:")
        for col in macro_present:
            non_null = df[col].notna().sum()
            pct = non_null / len(df) * 100
            print(f"  {col}: {non_null:,} values ({pct:.1f}% coverage)")
    else:
        print("WARNING: No macro features found in dataset")
        print("Strategy will use fundamentals only")
    
    # Run backtest - RESTRICTED UNIVERSE (Mid Cap+)
    print("\n" + "="*80)
    print("RUNNING BACKTEST - RESTRICTED UNIVERSE (Mid Cap and Above)")
    print("With Sector Neutralization + Rank-Based Targets + Macro Features")
    print("="*80)
    results_restricted, model_restricted = expanding_window_backtest(
        df, 
        portfolio_sizes=[10],
        start_year=2017,
        end_year=2023,
        min_market_cap='Mid Cap',
        reference_month=4,
        reference_day=30
    )
    
    # Calculate and display metrics for restricted
    print("\n" + "="*80)
    print("PERFORMANCE METRICS - RESTRICTED UNIVERSE (Mid Cap+)")
    print("="*80)
    metrics_restricted = calculate_performance_metrics(results_restricted)
    print(metrics_restricted.to_string(index=False))
    
    # Generate comprehensive reports for restricted universe
    print("\n" + "="*80)
    print("GENERATING DETAILED STOCK SELECTION REPORTS")
    print("="*80)
    print("\nRestricted Universe Reports:")
    report_restricted = generate_stock_selection_report(
        df,
        results_restricted,
        model_restricted,
        'restricted'
    )
    
    # Create comparison analysis
    print("\n" + "="*80)
    print("SELECTION ANALYSIS")
    print("="*80)
    
    print("\nTop 10 most selected tickers (Restricted):")
    print(report_restricted['TICKER'].value_counts().head(10))
    
    print("\nAverage characteristics of selected stocks:")
    print("\nRestricted Universe:")
    print(f"  Avg Market Cap: ${report_restricted['MthCap'].mean()/1e9:.2f}B")
    if 'bm' in report_restricted.columns:
        print(f"  Avg Book/Market: {report_restricted['bm'].mean():.3f}")
    if 'roe' in report_restricted.columns:
        print(f"  Avg ROE: {report_restricted['roe'].mean():.2%}")
    print(f"  Avg Actual Return: {report_restricted['Actual_1yr_Return'].mean():.2%}")
    
    # Create visualizations
    print("\nGenerating plots...")
    plot_results(results_restricted, save_path='results/figures/results_restricted_with_macro.png')
    plt.suptitle('Restricted Universe (Sector-Neutral + Rank-Based + Macro)', fontsize=16, y=1.00)
    
    print("\n" + "="*80)
    print("Backtest completed successfully!")
    print("="*80)
    print("\nKey features implemented:")
    print("  ✓ Sector neutralization (compares stocks within sectors)")
    print("  ✓ Rank-based targets (learns stock picking, not market timing)")
    print("  ✓ Target winsorization (handles outliers)")
    print("  ✓ Robust hyperparameters (prevents macro overfitting)")
    print("  ✓ MACRO FEATURES (global economic context, NOT neutralized)")