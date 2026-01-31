"""Tests for configuration module."""

import pytest
from src.config import (
    FEATURE_CATEGORIES,
    RF_PARAMS,
    MARKET_CAP_BOUNDARIES,
    IBKR_FEES,
    IRELAND_TAX,
    ALL_FEATURES,
)


def test_feature_categories_not_empty():
    """Test that feature categories are defined."""
    assert len(FEATURE_CATEGORIES) > 0
    for category, features in FEATURE_CATEGORIES.items():
        assert len(features) > 0, f"Category {category} is empty"


def test_all_features_flattened():
    """Test that ALL_FEATURES contains all category features."""
    expected_count = sum(len(f) for f in FEATURE_CATEGORIES.values())
    assert len(ALL_FEATURES) == expected_count


def test_rf_params_valid():
    """Test that RF params are valid for sklearn."""
    assert RF_PARAMS["n_estimators"] > 0
    assert 0 < RF_PARAMS["max_depth"] <= 100
    assert 0 < RF_PARAMS["max_features"] <= 1
    assert RF_PARAMS["min_samples_leaf"] > 0


def test_market_cap_boundaries_ordered():
    """Test that market cap boundaries are in ascending order."""
    values = list(MARKET_CAP_BOUNDARIES.values())
    assert values == sorted(values)


def test_ireland_tax_rates():
    """Test Ireland tax configuration."""
    assert IRELAND_TAX["cgt_rate"] == 0.33
    assert IRELAND_TAX["annual_exemption"] == 1270
    assert IRELAND_TAX["us_withholding_rate"] == 0.15


def test_ibkr_fees_structure():
    """Test IBKR fee configuration structure."""
    assert "tiered" in IBKR_FEES
    assert "fixed" in IBKR_FEES
    assert "us_stocks" in IBKR_FEES["tiered"]
    assert "per_share" in IBKR_FEES["tiered"]["us_stocks"]
