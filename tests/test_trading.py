"""Tests for trading modules."""

import pytest
import pandas as pd

from src.trading.fee_calculator import FeeCalculator
from src.trading.order_generator import OrderGenerator, ProposedOrder
from src.trading.position_manager import PositionManager


class TestFeeCalculator:
    """Tests for IBKR fee calculations."""

    def test_tiered_minimum_fee(self):
        """Test minimum fee enforcement."""
        calc = FeeCalculator(pricing_model="tiered")

        # Small trade should hit minimum
        fee = calc.calculate_fee(shares=10, price=50.00)

        # Minimum is $0.35
        assert fee >= 0.35

    def test_tiered_per_share(self):
        """Test per-share fee calculation."""
        calc = FeeCalculator(pricing_model="tiered")

        # Large trade: 1000 shares @ $0.0035 = $3.50
        fee = calc.calculate_fee(shares=1000, price=50.00, include_exchange=False)

        assert fee == pytest.approx(3.50)

    def test_max_percentage_cap(self):
        """Test 1% maximum fee cap."""
        calc = FeeCalculator(pricing_model="tiered")

        # Very large trade where per-share exceeds 1%
        # 10000 shares @ $1 = $10,000 trade value
        # Per-share: 10000 * 0.0035 = $35
        # 1% cap: 10000 * 0.01 = $100
        # Should use per-share since it's lower
        fee = calc.calculate_fee(shares=10000, price=1.00, include_exchange=False)

        assert fee <= 100  # 1% of trade value

    def test_fixed_pricing(self):
        """Test fixed pricing model."""
        calc = FeeCalculator(pricing_model="fixed")

        # Fixed: $0.005/share, min $1.00
        fee = calc.calculate_fee(shares=100, price=50.00, include_exchange=False)

        # 100 * 0.005 = 0.50, but min is $1.00
        assert fee == pytest.approx(1.00)

    def test_fee_breakdown(self):
        """Test detailed fee breakdown."""
        calc = FeeCalculator(pricing_model="tiered")

        breakdown = calc.calculate_fee_breakdown(shares=1000, price=50.00)

        assert breakdown.base_fee > 0
        assert breakdown.total > 0
        assert breakdown.fee_as_pct > 0

    def test_estimate_rebalance_cost(self):
        """Test rebalance cost estimation."""
        calc = FeeCalculator()

        trades = [
            (100, 50.00),  # Buy 100 @ $50
            (50, 100.00),  # Buy 50 @ $100
            (75, 80.00),   # Sell 75 @ $80
        ]

        cost = calc.estimate_rebalance_cost(trades)

        assert cost["trade_count"] == 3
        assert cost["total_fees"] > 0
        assert cost["total_traded_value"] > 0


class TestOrderGenerator:
    """Tests for order generation."""

    @pytest.fixture
    def generator(self):
        """Create an order generator."""
        return OrderGenerator()

    def test_generate_initial_orders(self, generator):
        """Test initial portfolio construction."""
        target = pd.DataFrame({
            "TICKER": ["AAPL", "MSFT", "GOOGL"],
            "predicted_rank": [0.9, 0.85, 0.8],
            "sector": ["Tech", "Tech", "Tech"],
        })

        prices = {"AAPL": 150.00, "MSFT": 300.00, "GOOGL": 100.00}

        orders = generator.generate_initial_orders(
            target_portfolio=target,
            account_value=30000,
            prices=prices,
        )

        assert len(orders) == 3
        assert all(o.action == "BUY" for o in orders)

        # Check equal weighting (approximately)
        total_value = sum(o.estimated_value for o in orders)
        assert total_value < 30000  # Should be less due to reserve

    def test_generate_rebalance_orders(self, generator):
        """Test rebalance order generation."""
        current = pd.DataFrame({
            "symbol": ["AAPL", "MSFT"],
            "shares": [100, 50],
        })

        target = pd.DataFrame({
            "TICKER": ["MSFT", "GOOGL"],
            "predicted_rank": [0.9, 0.85],
        })

        prices = {"AAPL": 150.00, "MSFT": 300.00, "GOOGL": 100.00}

        orders = generator.generate_rebalance_orders(
            current_positions=current,
            target_portfolio=target,
            account_value=30000,
            prices=prices,
        )

        # Should sell AAPL (exit), rebalance MSFT, buy GOOGL
        actions = {o.symbol: o.action for o in orders}

        assert actions.get("AAPL") == "SELL"
        assert "GOOGL" in actions

    def test_proposed_order_net_value(self):
        """Test net value calculation."""
        buy_order = ProposedOrder(
            symbol="AAPL",
            action="BUY",
            quantity=100,
            estimated_price=150.00,
            estimated_fee=1.00,
        )

        # Net value for buy is negative (cash outflow)
        assert buy_order.net_value < 0

        sell_order = ProposedOrder(
            symbol="AAPL",
            action="SELL",
            quantity=100,
            estimated_price=150.00,
            estimated_fee=1.00,
        )

        # Net value for sell is positive (cash inflow)
        assert sell_order.net_value > 0


class TestPositionManager:
    """Tests for position management."""

    @pytest.fixture
    def manager(self, tmp_path):
        """Create a position manager with temp directory."""
        return PositionManager(portfolio_dir=tmp_path)

    def test_add_position(self, manager):
        """Test adding a new position."""
        pos = manager.add_position("AAPL", shares=100, price=150.00)

        assert pos.symbol == "AAPL"
        assert pos.shares == 100
        assert pos.avg_cost == 150.00
        assert pos.total_cost == 15000.00

    def test_add_to_existing_position(self, manager):
        """Test averaging into a position."""
        manager.add_position("AAPL", shares=100, price=100.00)
        pos = manager.add_position("AAPL", shares=100, price=150.00)

        assert pos.shares == 200
        # Average cost: (100*100 + 100*150) / 200 = 125
        assert pos.avg_cost == pytest.approx(125.00)

    def test_reduce_position(self, manager):
        """Test reducing a position."""
        manager.add_position("AAPL", shares=100, price=150.00)
        pos = manager.reduce_position("AAPL", shares=50, price=175.00)

        assert pos.shares == 50
        # Average cost unchanged
        assert pos.avg_cost == 150.00

    def test_close_position(self, manager):
        """Test closing entire position."""
        manager.add_position("AAPL", shares=100, price=150.00)
        result = manager.reduce_position("AAPL", shares=100, price=175.00)

        assert result is None
        assert "AAPL" not in manager.positions

    def test_update_prices(self, manager):
        """Test price update and P&L calculation."""
        manager.add_position("AAPL", shares=100, price=100.00)
        manager.update_prices({"AAPL": 150.00})

        pos = manager.get_position("AAPL")

        assert pos.current_price == 150.00
        assert pos.market_value == 15000.00
        assert pos.unrealized_pnl == 5000.00  # 150-100 * 100

    def test_get_summary(self, manager):
        """Test portfolio summary."""
        manager.add_position("AAPL", shares=100, price=100.00)
        manager.add_position("MSFT", shares=50, price=200.00)
        manager.cash = 5000.00

        summary = manager.get_summary()

        assert summary["n_positions"] == 2
        assert summary["cash"] == 5000.00
        assert summary["invested"] == 20000.00  # 10000 + 10000
        assert summary["total_value"] == 25000.00

    def test_save_and_load_snapshot(self, manager, tmp_path):
        """Test portfolio persistence."""
        manager.add_position("AAPL", shares=100, price=150.00)
        manager.cash = 5000.00

        filepath = manager.save_snapshot("test")

        new_manager = PositionManager(portfolio_dir=tmp_path)
        new_manager.load_snapshot(filepath)

        assert new_manager.cash == 5000.00
        assert "AAPL" in new_manager.positions
        assert new_manager.positions["AAPL"].shares == 100
