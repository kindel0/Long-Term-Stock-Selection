"""
Unified Random Forest model for stock selection.

This is the single source of truth for the RF model, consolidating
previously duplicated implementations across multiple scripts.

Based on Wynne (2023) thesis methodology.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from ..config import (
    RF_PARAMS,
    FEATURE_CATEGORIES,
    NON_NEUTRALIZED_FEATURES,
    MISSING_THRESHOLD_PCT,
    ROW_COMPLETENESS_THRESHOLD,
    WINSORIZE_PERCENTILES,
    EPS,
)

logger = logging.getLogger(__name__)


class StockSelectionRF:
    """
    Random Forest model for long-term stock selection.

    Features:
    - Sector neutralization (fundamentals only, not macro)
    - Rank-based targets (prevents market timing bias)
    - Consistent hyperparameters from central config
    - Proper index alignment across train/test splits
    - Handles missing data with configurable thresholds

    Attributes:
        model: The underlying RandomForestRegressor
        feature_columns: List of feature columns after preprocessing
        feature_importances: DataFrame with feature importance scores
        feature_medians_: Median values for imputation
        sector_stats_: Sector statistics for neutralization
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """
        Initialize the stock selection model.

        Args:
            params: Optional dict of RF hyperparameters. If None, uses RF_PARAMS from config.
        """
        self.params = params or RF_PARAMS.copy()
        self.model = RandomForestRegressor(**self.params)

        self.feature_columns: Optional[List[str]] = None
        self.feature_importances: Optional[pd.DataFrame] = None
        self.feature_medians_: Optional[pd.Series] = None
        self.sector_stats_: Optional[Dict] = None

        # Features that should not be sector-neutralized
        self._non_neutralized = set(NON_NEUTRALIZED_FEATURES)

    def prepare_features(self, df: pd.DataFrame) -> List[str]:
        """
        Get list of available features from the DataFrame.

        Args:
            df: DataFrame containing potential feature columns

        Returns:
            List of feature column names present in df
        """
        all_features = []
        for category, features in FEATURE_CATEGORIES.items():
            all_features.extend(features)

        available = [col for col in all_features if col in df.columns]
        logger.debug(f"Found {len(available)} of {len(all_features)} features")
        return available

    def neutralize_features(
        self, X: pd.DataFrame, metadata: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Z-score normalize fundamental features by sector and date.

        Macro features and size are passed through raw (not neutralized).
        This removes sector effects from fundamentals while preserving
        macro timing signals.

        Args:
            X: Feature DataFrame
            metadata: DataFrame with 'sector' and 'public_date' columns

        Returns:
            Neutralized feature DataFrame with same index as X
        """
        if "sector" not in metadata.columns:
            logger.warning("No 'sector' column found, skipping neutralization")
            return X

        X_neutral = X.copy()

        # Combine for groupby operation
        combined = X.copy()
        combined["sector"] = metadata["sector"].values
        combined["public_date"] = metadata["public_date"].values

        for col in X.columns:
            if col in self._non_neutralized:
                # Pass macro/size features through unchanged
                continue

            # Z-score by sector and date
            grouped = combined.groupby(["sector", "public_date"])[col]
            group_mean = grouped.transform("mean")
            group_std = grouped.transform("std")

            # Avoid division by zero
            z_scores = (combined[col] - group_mean) / (group_std + EPS)

            # Handle infinities and NaN from constant groups
            z_scores = z_scores.replace([np.inf, -np.inf], np.nan)
            z_scores = z_scores.fillna(0)

            X_neutral[col] = z_scores

        return X_neutral

    def winsorize_target(
        self, y: pd.Series, lower_pct: float = None, upper_pct: float = None
    ) -> pd.Series:
        """
        Winsorize target variable to reduce impact of outliers.

        Args:
            y: Target series
            lower_pct: Lower percentile (default from config)
            upper_pct: Upper percentile (default from config)

        Returns:
            Winsorized target series
        """
        lower_pct = lower_pct or WINSORIZE_PERCENTILES[0]
        upper_pct = upper_pct or WINSORIZE_PERCENTILES[1]

        valid_y = y.dropna()
        if len(valid_y) == 0:
            return y

        lower_bound = np.percentile(valid_y, lower_pct)
        upper_bound = np.percentile(valid_y, upper_pct)

        return y.clip(lower=lower_bound, upper=upper_bound)

    def rank_target(self, y: pd.Series, dates: pd.Series) -> pd.Series:
        """
        Convert target to cross-sectional ranks by date.

        This prevents the model from learning market timing
        and focuses on relative stock selection.

        Args:
            y: Target returns
            dates: Corresponding dates for grouping

        Returns:
            Percentile ranks (0-1) by date
        """
        temp_df = pd.DataFrame({"target": y, "date": dates}, index=y.index)
        temp_df["rank"] = temp_df.groupby("date")["target"].rank(pct=True)
        return temp_df["rank"]

    def handle_missing_data(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        metadata: Optional[pd.DataFrame] = None,
        missing_threshold: float = None,
        row_completeness: float = None,
    ) -> Tuple[pd.DataFrame, Optional[pd.Series], Optional[pd.DataFrame], List[str]]:
        """
        Handle missing data with proper index alignment.

        This method:
        1. Drops features with too many missing values
        2. Drops rows that are too incomplete
        3. Imputes remaining missing values with medians
        4. Returns aligned X, y, and metadata

        Args:
            X: Feature DataFrame
            y: Optional target Series (will be aligned)
            metadata: Optional metadata DataFrame (will be aligned)
            missing_threshold: Max fraction of missing per feature (default from config)
            row_completeness: Min fraction of non-null per row (default from config)

        Returns:
            Tuple of (X_clean, y_aligned, metadata_aligned, dropped_features)
        """
        missing_threshold = missing_threshold or MISSING_THRESHOLD_PCT
        row_completeness = row_completeness or ROW_COMPLETENESS_THRESHOLD

        initial_features = X.shape[1]
        initial_rows = len(X)

        # Step 1: Drop high-missing features
        missing_pct = X.isnull().sum() / len(X)
        features_to_keep = missing_pct[missing_pct <= missing_threshold].index.tolist()
        dropped_features = missing_pct[missing_pct > missing_threshold].index.tolist()

        if dropped_features:
            logger.info(
                f"Dropping {len(dropped_features)} features with "
                f">{missing_threshold*100:.0f}% missing: {dropped_features[:5]}..."
            )

        X_filtered = X[features_to_keep].copy()

        # Step 2: Drop incomplete rows
        completeness = X_filtered.notna().sum(axis=1) / X_filtered.shape[1]
        rows_to_keep = completeness >= row_completeness
        X_filtered = X_filtered[rows_to_keep].copy()

        kept_indices = X_filtered.index

        logger.info(
            f"Kept {len(X_filtered)} of {initial_rows} rows "
            f"({len(X_filtered)/initial_rows*100:.1f}%)"
        )

        # Step 3: Compute and store medians for imputation
        if self.feature_medians_ is None:
            self.feature_medians_ = X_filtered.median()

        # Step 4: Impute remaining missing values
        X_clean = X_filtered.fillna(self.feature_medians_)

        # Handle any remaining NaN (e.g., columns that were all NaN)
        if X_clean.isnull().sum().sum() > 0:
            X_clean = X_clean.fillna(0)

        # Step 5: Align y and metadata to the same indices
        y_aligned = None
        if y is not None:
            y_aligned = y.loc[kept_indices].copy()

        metadata_aligned = None
        if metadata is not None:
            metadata_aligned = metadata.loc[kept_indices].copy()

        return X_clean, y_aligned, metadata_aligned, dropped_features

    def train(
        self, X: pd.DataFrame, y: pd.Series, metadata: pd.DataFrame
    ) -> "StockSelectionRF":
        """
        Train the model on the provided data.

        Args:
            X: Feature DataFrame (will be preprocessed)
            y: Target returns
            metadata: DataFrame with 'sector' and 'public_date'

        Returns:
            self for method chaining
        """
        # Filter to valid target rows
        valid_mask = y.notna()
        X = X[valid_mask].copy()
        y = y[valid_mask].copy()
        metadata = metadata[valid_mask].copy()

        logger.info(f"Training RF on {len(X)} samples with {X.shape[1]} features")

        # Handle missing data (with alignment)
        X_clean, y_clean, meta_clean, dropped = self.handle_missing_data(
            X, y, metadata
        )

        # Neutralize features
        X_neutral = self.neutralize_features(X_clean, meta_clean)

        # Transform target
        y_winsorized = self.winsorize_target(y_clean)
        y_ranked = self.rank_target(y_winsorized, meta_clean["public_date"])

        # Store feature columns
        self.feature_columns = X_neutral.columns.tolist()

        # Fit the model
        self.model.fit(X_neutral, y_ranked)

        # Store feature importances
        self.feature_importances = pd.DataFrame(
            {"feature": self.feature_columns, "importance": self.model.feature_importances_}
        ).sort_values("importance", ascending=False)

        logger.info(
            f"Top 5 features: {self.feature_importances.head()['feature'].tolist()}"
        )

        return self

    def predict(
        self, X: pd.DataFrame, metadata: Optional[pd.DataFrame] = None
    ) -> np.ndarray:
        """
        Predict rankings for new data.

        Args:
            X: Feature DataFrame
            metadata: Optional metadata for neutralization

        Returns:
            Predicted rankings as numpy array
        """
        if self.feature_columns is None:
            raise ValueError("Model has not been trained yet")

        # Handle missing features
        X_test = X.copy()
        for col in self.feature_columns:
            if col not in X_test.columns:
                X_test[col] = 0

        X_test = X_test[self.feature_columns]

        # Impute missing values
        if self.feature_medians_ is not None:
            X_test = X_test.fillna(self.feature_medians_)
        X_test = X_test.fillna(0)

        # Neutralize if metadata provided
        if metadata is not None:
            X_test = self.neutralize_features(X_test, metadata)

        return self.model.predict(X_test)

    def select_stocks(
        self,
        X: pd.DataFrame,
        metadata: pd.DataFrame,
        n: int = 15,
        additional_data: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Select top n stocks based on predicted rankings.

        Args:
            X: Feature DataFrame
            metadata: Metadata with sector, date, ticker info
            n: Number of stocks to select
            additional_data: Optional DataFrame with additional columns to include

        Returns:
            DataFrame with selected stocks and their details
        """
        predictions = self.predict(X, metadata)

        results = pd.DataFrame(
            {"predicted_rank": predictions, "TICKER": metadata.get("TICKER", X.index)},
            index=X.index,
        )

        # Add metadata columns
        for col in ["sector", "public_date", "MthCap"]:
            if col in metadata.columns:
                results[col] = metadata[col].values

        # Add additional columns if provided
        if additional_data is not None:
            for col in additional_data.columns:
                if col not in results.columns:
                    results[col] = additional_data.loc[X.index, col].values

        # Select top n
        top_stocks = results.nlargest(n, "predicted_rank")

        return top_stocks

    def get_feature_importances(self) -> pd.DataFrame:
        """
        Get feature importances from the trained model.

        Returns:
            DataFrame with feature names and importance scores
        """
        if self.feature_importances is None:
            raise ValueError("Model has not been trained yet")
        return self.feature_importances.copy()

    def save(self, path: str) -> None:
        """Save the trained model to disk."""
        import joblib

        joblib.dump(
            {
                "model": self.model,
                "feature_columns": self.feature_columns,
                "feature_medians": self.feature_medians_,
                "feature_importances": self.feature_importances,
                "params": self.params,
            },
            path,
        )
        logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str) -> "StockSelectionRF":
        """Load a trained model from disk."""
        import joblib

        data = joblib.load(path)

        instance = cls(params=data["params"])
        instance.model = data["model"]
        instance.feature_columns = data["feature_columns"]
        instance.feature_medians_ = data["feature_medians"]
        instance.feature_importances = data["feature_importances"]

        logger.info(f"Model loaded from {path}")
        return instance
