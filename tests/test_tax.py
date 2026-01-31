"""Tests for tax calculation modules."""

from datetime import datetime
import pytest

from src.tax.cost_basis import CostBasisTracker, TaxLot
from src.tax.ireland_cgt import IrelandCGTCalculator, TaxableGain
from src.tax.dividend_tracker import DividendTracker


class TestCostBasisTracker:
    """Tests for FIFO cost basis tracking."""

    @pytest.fixture
    def tracker(self, tmp_path):
        """Create a tracker with temp directory."""
        return CostBasisTracker(data_dir=tmp_path)

    def test_record_purchase(self, tracker):
        """Test recording a purchase."""
        lot = tracker.record_purchase(
            symbol="AAPL",
            shares=100,
            price=150.00,
            date=datetime(2024, 1, 15),
            fees=1.00,
        )

        assert lot.symbol == "AAPL"
        assert lot.shares == 100
        assert lot.remaining_shares == 100
        assert lot.cost_per_share == 150.00
        assert lot.total_cost == 15000.00

    def test_fifo_sale(self, tracker):
        """Test FIFO cost basis calculation."""
        # Buy at different prices
        tracker.record_purchase("AAPL", 100, 100.00, datetime(2024, 1, 1))
        tracker.record_purchase("AAPL", 100, 150.00, datetime(2024, 2, 1))

        # Sell 150 shares - should use oldest first
        result = tracker.record_sale("AAPL", 150, 175.00, datetime(2024, 3, 1))

        # Cost basis: 100 shares @ $100 + 50 shares @ $150 = $17,500
        assert result.total_cost_basis == pytest.approx(100 * 100 + 50 * 150)

        # Proceeds: 150 * $175 = $26,250
        assert result.total_proceeds == pytest.approx(150 * 175)

        # Gain: $26,250 - $17,500 = $8,750
        assert result.total_gain == pytest.approx(8750)

        # Remaining: 50 shares from second lot
        assert tracker.get_total_shares("AAPL") == 50

    def test_insufficient_shares(self, tracker):
        """Test selling more shares than available."""
        tracker.record_purchase("AAPL", 100, 150.00, datetime(2024, 1, 1))

        with pytest.raises(ValueError):
            tracker.record_sale("AAPL", 150, 175.00, datetime(2024, 2, 1))

    def test_average_cost(self, tracker):
        """Test average cost calculation."""
        tracker.record_purchase("AAPL", 100, 100.00, datetime(2024, 1, 1))
        tracker.record_purchase("AAPL", 100, 150.00, datetime(2024, 2, 1))

        avg = tracker.get_average_cost("AAPL")
        # (100*100 + 100*150) / 200 = 125
        assert avg == pytest.approx(125.00)

    def test_save_and_load(self, tracker, tmp_path):
        """Test persistence."""
        tracker.record_purchase("AAPL", 100, 150.00, datetime(2024, 1, 1))

        filepath = tracker.save("test_basis.json")
        assert filepath.exists()

        # Load in new tracker
        new_tracker = CostBasisTracker(data_dir=tmp_path)
        new_tracker.load("test_basis.json")

        assert new_tracker.get_total_shares("AAPL") == 100


class TestIrelandCGTCalculator:
    """Tests for Ireland CGT calculations."""

    @pytest.fixture
    def calculator(self, tmp_path):
        """Create a CGT calculator with temp directory."""
        return IrelandCGTCalculator(data_dir=tmp_path)

    def test_record_gain(self, calculator):
        """Test recording a gain."""
        calculator.record_purchase("AAPL", 100, 100.00, datetime(2024, 1, 1))
        gain = calculator.record_sale("AAPL", 100, 150.00, datetime(2024, 6, 1))

        assert not gain.is_loss
        assert gain.gross_gain == pytest.approx(5000.00)

    def test_record_loss(self, calculator):
        """Test recording a loss."""
        calculator.record_purchase("AAPL", 100, 150.00, datetime(2024, 1, 1))
        gain = calculator.record_sale("AAPL", 100, 100.00, datetime(2024, 6, 1))

        assert gain.is_loss
        assert gain.gross_gain == pytest.approx(-5000.00)

    def test_annual_summary(self, calculator):
        """Test annual CGT summary calculation."""
        # Gain
        calculator.record_purchase("AAPL", 100, 100.00, datetime(2024, 1, 1))
        calculator.record_sale("AAPL", 100, 150.00, datetime(2024, 6, 1))  # +5000

        # Loss
        calculator.record_purchase("MSFT", 50, 200.00, datetime(2024, 2, 1))
        calculator.record_sale("MSFT", 50, 150.00, datetime(2024, 7, 1))  # -2500

        summary = calculator.get_annual_summary(2024)

        assert summary.gross_gains == pytest.approx(5000.00)
        assert summary.gross_losses == pytest.approx(2500.00)
        assert summary.net_gain == pytest.approx(2500.00)

        # Taxable = max(0, net_gain - exemption) = max(0, 2500 - 1270) = 1230
        assert summary.taxable_gain == pytest.approx(1230.00)

        # Tax = 1230 * 0.33 = 405.90
        assert summary.estimated_tax == pytest.approx(405.90)

    def test_exemption_covers_gain(self, calculator):
        """Test when exemption covers entire gain."""
        calculator.record_purchase("AAPL", 10, 100.00, datetime(2024, 1, 1))
        calculator.record_sale("AAPL", 10, 150.00, datetime(2024, 6, 1))  # +500

        summary = calculator.get_annual_summary(2024)

        # Gain of 500 is less than 1270 exemption
        assert summary.taxable_gain == 0
        assert summary.estimated_tax == 0

    def test_estimate_tax_on_sale(self, calculator):
        """Test tax estimation before sale."""
        calculator.record_purchase("AAPL", 100, 100.00, datetime(2024, 1, 1))

        estimate = calculator.estimate_tax_on_sale("AAPL", 50, 150.00)

        assert estimate["estimated_proceeds"] == pytest.approx(7500.00)
        assert estimate["estimated_cost_basis"] == pytest.approx(5000.00)
        assert estimate["gross_gain"] == pytest.approx(2500.00)


class TestDividendTracker:
    """Tests for dividend tracking."""

    @pytest.fixture
    def tracker(self, tmp_path):
        """Create a dividend tracker with temp directory."""
        return DividendTracker(data_dir=tmp_path)

    def test_record_dividend(self, tracker):
        """Test recording a dividend."""
        record = tracker.record_dividend(
            symbol="AAPL",
            payment_date=datetime(2024, 3, 15),
            shares=100,
            dividend_per_share=0.25,
        )

        assert record.gross_amount == pytest.approx(25.00)
        assert record.withholding_rate == pytest.approx(0.15)
        assert record.withholding_amount == pytest.approx(3.75)
        assert record.net_amount == pytest.approx(21.25)

    def test_annual_summary(self, tracker):
        """Test annual dividend summary."""
        tracker.record_dividend("AAPL", datetime(2024, 3, 15), 100, 0.25)
        tracker.record_dividend("AAPL", datetime(2024, 6, 15), 100, 0.25)
        tracker.record_dividend("MSFT", datetime(2024, 6, 20), 50, 0.50)

        summary = tracker.get_annual_summary(2024)

        # Total gross: 25 + 25 + 25 = 75
        assert summary.total_gross == pytest.approx(75.00)

        # Total withholding: 75 * 0.15 = 11.25
        assert summary.total_withholding == pytest.approx(11.25)

        # Net: 75 - 11.25 = 63.75
        assert summary.total_net == pytest.approx(63.75)

        assert summary.n_payments == 3

    def test_record_from_broker(self, tracker):
        """Test recording from broker statement."""
        record = tracker.record_from_broker(
            symbol="AAPL",
            payment_date=datetime(2024, 3, 15),
            gross_amount=100.00,
            net_amount=85.00,
        )

        assert record.gross_amount == 100.00
        assert record.net_amount == 85.00
        assert record.withholding_amount == 15.00
        assert record.withholding_rate == pytest.approx(0.15)

    def test_save_and_load(self, tracker, tmp_path):
        """Test persistence."""
        tracker.record_dividend("AAPL", datetime(2024, 3, 15), 100, 0.25)

        filepath = tracker.save("test_dividends.json")
        assert filepath.exists()

        new_tracker = DividendTracker(data_dir=tmp_path)
        new_tracker.load("test_dividends.json")

        assert len(new_tracker.records) == 1
