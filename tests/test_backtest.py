"""Tests for backtest modules."""

import numpy as np
import pandas as pd
import pytest

from src.backtest.metrics import calculate_metrics, PerformanceMetrics


class TestPerformanceMetrics:
    """Tests for performance metric calculations."""

    @pytest.fixture
    def sample_returns(self):
        """Create sample return series."""
        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=20, freq="QE")
        returns = pd.Series(np.random.randn(20) * 0.05 + 0.02, index=dates)
        return returns

    @pytest.fixture
    def sample_benchmark(self):
        """Create sample benchmark returns."""
        np.random.seed(123)
        dates = pd.date_range("2020-01-01", periods=20, freq="QE")
        returns = pd.Series(np.random.randn(20) * 0.04 + 0.015, index=dates)
        return returns

    def test_basic_metrics(self, sample_returns):
        """Test basic return metrics."""
        metrics = calculate_metrics(sample_returns, periods_per_year=4)

        assert metrics.n_periods == 20
        assert metrics.periods_per_year == 4
        assert metrics.avg_period_return != 0
        assert metrics.best_period >= metrics.worst_period

    def test_total_return(self, sample_returns):
        """Test total return calculation."""
        metrics = calculate_metrics(sample_returns, periods_per_year=4)

        # Verify compounding
        expected = (1 + sample_returns).prod() - 1
        assert metrics.total_return == pytest.approx(expected, rel=0.01)

    def test_volatility(self, sample_returns):
        """Test volatility calculation."""
        metrics = calculate_metrics(sample_returns, periods_per_year=4)

        # Annualized volatility
        expected = sample_returns.std() * np.sqrt(4)
        assert metrics.volatility == pytest.approx(expected, rel=0.01)

    def test_max_drawdown(self, sample_returns):
        """Test max drawdown calculation."""
        metrics = calculate_metrics(sample_returns, periods_per_year=4)

        # Max drawdown should be positive (magnitude)
        assert metrics.max_drawdown >= 0

    def test_sharpe_ratio(self, sample_returns):
        """Test Sharpe ratio calculation."""
        metrics = calculate_metrics(sample_returns, risk_free_rate=0.02, periods_per_year=4)

        # Sharpe should be reasonable for random returns
        assert -5 < metrics.sharpe_ratio < 5

    def test_benchmark_comparison(self, sample_returns, sample_benchmark):
        """Test benchmark comparison metrics."""
        metrics = calculate_metrics(
            sample_returns,
            benchmark_returns=sample_benchmark,
            periods_per_year=4,
        )

        # Should have alpha and beta
        assert metrics.alpha != 0 or metrics.beta != 0

        # Win rate should be between 0 and 1
        assert 0 <= metrics.win_rate <= 1

    def test_empty_returns(self):
        """Test handling of empty returns."""
        metrics = calculate_metrics(pd.Series([]), periods_per_year=4)

        assert metrics.n_periods == 0
        assert metrics.total_return == 0
        assert metrics.sharpe_ratio == 0

    def test_to_dict(self, sample_returns):
        """Test dictionary conversion."""
        metrics = calculate_metrics(sample_returns, periods_per_year=4)
        d = metrics.to_dict()

        assert "total_return_pct" in d
        assert "sharpe_ratio" in d
        assert isinstance(d["total_return_pct"], float)


class TestBenchmarkManager:
    """Tests for benchmark data management."""

    # Note: These tests require network access
    # Mark them to skip in CI if needed

    @pytest.mark.skip(reason="Requires network access")
    def test_fetch_data(self):
        """Test fetching benchmark data."""
        from src.backtest.benchmark import BenchmarkManager
        from datetime import datetime

        bm = BenchmarkManager()
        data = bm.fetch_data(
            start=datetime(2023, 1, 1),
            end=datetime(2023, 12, 31),
        )

        assert not data.empty
        assert "Adj Close" in data.columns

    @pytest.mark.skip(reason="Requires network access")
    def test_get_return(self):
        """Test return calculation between dates."""
        from src.backtest.benchmark import BenchmarkManager
        from datetime import datetime

        bm = BenchmarkManager()
        ret = bm.get_return(
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 3, 31),
        )

        assert ret is not None
        assert -1 < ret < 2  # Reasonable return range
