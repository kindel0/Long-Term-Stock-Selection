"""Tests for stock selection models."""

import numpy as np
import pandas as pd
import pytest

from src.models.stock_selection_rf import StockSelectionRF
from src.models.feature_engineering import FeatureEngineer


class TestStockSelectionRF:
    """Tests for the StockSelectionRF model."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        n_samples = 1000

        data = {
            "TICKER": [f"STOCK{i % 50}" for i in range(n_samples)],
            "public_date": pd.date_range("2020-01-01", periods=n_samples // 50, freq="ME").repeat(50),
            "sector": [f"Sector{i % 5}" for i in range(n_samples)],
            "bm": np.random.randn(n_samples),
            "roe": np.random.randn(n_samples),
            "roa": np.random.randn(n_samples),
            "de_ratio": np.random.randn(n_samples),
            "curr_ratio": np.random.randn(n_samples),
            "MthCap": np.random.uniform(1e9, 100e9, n_samples),
            "1yr_return": np.random.randn(n_samples) * 0.3,
        }

        return pd.DataFrame(data)

    def test_prepare_features(self, sample_data):
        """Test feature preparation."""
        model = StockSelectionRF()
        features = model.prepare_features(sample_data)

        assert len(features) > 0
        assert "bm" in features
        assert "roe" in features

    def test_neutralize_features(self, sample_data):
        """Test sector neutralization."""
        model = StockSelectionRF()

        features = ["bm", "roe", "roa"]
        X = sample_data[features].copy()
        meta = sample_data[["sector", "public_date"]].copy()

        X_neutral = model.neutralize_features(X, meta)

        assert X_neutral.shape == X.shape
        assert not X_neutral.isnull().all().any()

    def test_handle_missing_data(self, sample_data):
        """Test missing data handling."""
        model = StockSelectionRF()

        # Add missing values
        sample_data_with_missing = sample_data.copy()
        sample_data_with_missing.loc[:100, "bm"] = np.nan

        features = ["bm", "roe", "roa"]
        X = sample_data_with_missing[features].copy()
        y = sample_data_with_missing["1yr_return"].copy()
        meta = sample_data_with_missing[["sector", "public_date"]].copy()

        X_clean, y_clean, meta_clean, dropped = model.handle_missing_data(X, y, meta)

        assert not X_clean.isnull().any().any()
        assert len(X_clean) == len(y_clean)
        assert len(X_clean) == len(meta_clean)

    def test_train_and_predict(self, sample_data):
        """Test model training and prediction."""
        model = StockSelectionRF()

        features = model.prepare_features(sample_data)
        X = sample_data[features].copy()
        y = sample_data["1yr_return"].copy()
        meta = sample_data[["sector", "public_date", "TICKER"]].copy()

        # Train
        model.train(X, y, meta)

        assert model.feature_columns is not None
        assert model.feature_importances is not None

        # Predict
        predictions = model.predict(X, meta)

        assert len(predictions) == len(X)
        assert not np.isnan(predictions).any()

    def test_select_stocks(self, sample_data):
        """Test stock selection."""
        model = StockSelectionRF()

        features = model.prepare_features(sample_data)
        X = sample_data[features].copy()
        y = sample_data["1yr_return"].copy()
        meta = sample_data[["sector", "public_date", "TICKER", "MthCap"]].copy()

        model.train(X, y, meta)

        # Select on latest date
        latest_date = sample_data["public_date"].max()
        test_mask = sample_data["public_date"] == latest_date
        X_test = X[test_mask]
        meta_test = meta[test_mask]

        selected = model.select_stocks(X_test, meta_test, n=10)

        assert len(selected) == 10
        assert "predicted_rank" in selected.columns


class TestFeatureEngineer:
    """Tests for feature engineering."""

    @pytest.fixture
    def sample_flow_data(self):
        """Create sample flow data for TTM testing."""
        np.random.seed(42)

        data = {
            "TICKER": ["AAPL"] * 8,
            "Revenue": [100, 110, 105, 115, 120, 125, 130, 135],
            "Net Income": [10, 11, 10.5, 11.5, 12, 12.5, 13, 13.5],
        }

        return pd.DataFrame(data)

    def test_compute_ttm(self, sample_flow_data):
        """Test TTM calculation."""
        engineer = FeatureEngineer()

        result = engineer.compute_ttm(
            sample_flow_data,
            ticker_col="TICKER",
            flow_columns=["Revenue", "Net Income"],
            periods=4,
        )

        assert "Revenue_ttm" in result.columns
        assert "Net Income_ttm" in result.columns

        # First 3 rows should be NaN (need 4 periods)
        assert result["Revenue_ttm"].iloc[:3].isna().all()

        # Fourth row should be sum of first 4
        expected_ttm = sum(sample_flow_data["Revenue"].iloc[:4])
        assert result["Revenue_ttm"].iloc[3] == pytest.approx(expected_ttm)

    def test_winsorize_ratios(self):
        """Test ratio winsorization."""
        engineer = FeatureEngineer(winsorize_mad=3)

        data = pd.DataFrame({
            "ratio1": [1, 2, 3, 4, 5, 100],  # 100 is outlier
        })

        result = engineer.winsorize_ratios(data, columns=["ratio1"])

        # Outlier should be clipped
        assert result["ratio1"].max() < 100

    def test_calculate_forward_returns(self):
        """Test forward return calculation."""
        engineer = FeatureEngineer()

        data = pd.DataFrame({
            "TICKER": ["AAPL"] * 6,
            "public_date": pd.date_range("2020-01-31", periods=6, freq="ME"),
            "MthPrc": [100, 110, 105, 120, 115, 130],
        })

        result = engineer.calculate_forward_returns(
            data, price_col="MthPrc", periods=3, validate_gap=False
        )

        assert "3mo_return" in result.columns
        # First return: (120 - 100) / 100 = 0.20
        assert result["3mo_return"].iloc[0] == pytest.approx(0.20, rel=0.01)
