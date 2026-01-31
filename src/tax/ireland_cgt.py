"""
Ireland Capital Gains Tax calculator.

Implements Ireland CGT rules for stock trading:
- 33% CGT rate
- EUR 1,270 annual exemption
- Losses can offset gains
- Same-day and 4-week matching rules (simplified)
"""

import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from .cost_basis import CostBasisTracker, SaleResult
from ..config import IRELAND_TAX, TRADES_DIR

logger = logging.getLogger(__name__)


@dataclass
class TaxableGain:
    """Details of a taxable gain or loss from a sale."""

    symbol: str
    sale_date: datetime
    shares_sold: int
    sale_proceeds: float
    cost_basis: float
    gross_gain: float
    is_loss: bool
    holding_period_days: int
    notes: str = ""

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            **asdict(self),
            "sale_date": self.sale_date.isoformat(),
        }


@dataclass
class AnnualCGTSummary:
    """Annual CGT summary for tax return."""

    year: int
    total_proceeds: float
    total_cost_basis: float
    gross_gains: float
    gross_losses: float
    net_gain: float
    annual_exemption: float
    taxable_gain: float
    cgt_rate: float
    estimated_tax: float
    n_disposals: int
    gains: List[TaxableGain]
    losses: List[TaxableGain]


class IrelandCGTCalculator:
    """
    Ireland Capital Gains Tax calculator.

    Tracks sales throughout the year and calculates CGT liability.
    Uses FIFO cost basis via CostBasisTracker.

    Example:
        calc = IrelandCGTCalculator()
        calc.record_purchase("AAPL", 100, 150.00, datetime.now(), 1.00)
        gain = calc.record_sale("AAPL", 50, 175.00, datetime.now(), 0.50)
        summary = calc.get_annual_summary(2024)
    """

    def __init__(
        self,
        cost_tracker: Optional[CostBasisTracker] = None,
        data_dir: Optional[Path] = None,
    ):
        """
        Initialize the CGT calculator.

        Args:
            cost_tracker: Optional cost basis tracker (creates one if not provided)
            data_dir: Directory for persisting data
        """
        self.cost_tracker = cost_tracker or CostBasisTracker()
        self.data_dir = Path(data_dir) if data_dir else TRADES_DIR
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.cgt_rate = IRELAND_TAX["cgt_rate"]
        self.annual_exemption = IRELAND_TAX["annual_exemption"]

        # Track all disposals for tax reporting
        self.disposals: List[TaxableGain] = []

    def record_purchase(
        self,
        symbol: str,
        shares: int,
        price: float,
        date: datetime,
        fees: float = 0.0,
    ) -> None:
        """
        Record a purchase.

        Args:
            symbol: Ticker symbol
            shares: Number of shares
            price: Price per share
            date: Purchase date
            fees: Transaction fees
        """
        self.cost_tracker.record_purchase(symbol, shares, price, date, fees)

    def record_sale(
        self,
        symbol: str,
        shares: int,
        price: float,
        date: datetime,
        fees: float = 0.0,
    ) -> TaxableGain:
        """
        Record a sale and calculate gain/loss.

        Args:
            symbol: Ticker symbol
            shares: Number of shares sold
            price: Sale price per share
            date: Sale date
            fees: Transaction fees

        Returns:
            TaxableGain with details
        """
        # Get cost basis and lots used
        sale_result = self.cost_tracker.record_sale(
            symbol, shares, price, date, fees
        )

        # Calculate holding period (using oldest lot)
        oldest_lot_date = None
        for lot_id, _, _ in sale_result.lots_used:
            # Extract date from lot_id (format: SYMBOL-YYYYMMDDHHMMSS)
            try:
                date_str = lot_id.split("-")[1]
                lot_date = datetime.strptime(date_str, "%Y%m%d%H%M%S")
                if oldest_lot_date is None or lot_date < oldest_lot_date:
                    oldest_lot_date = lot_date
            except (IndexError, ValueError):
                pass

        holding_days = (date - oldest_lot_date).days if oldest_lot_date else 0

        gain = TaxableGain(
            symbol=symbol,
            sale_date=date,
            shares_sold=shares,
            sale_proceeds=sale_result.total_proceeds,
            cost_basis=sale_result.total_cost_basis,
            gross_gain=sale_result.total_gain,
            is_loss=sale_result.total_gain < 0,
            holding_period_days=holding_days,
        )

        self.disposals.append(gain)
        logger.info(
            f"Recorded disposal: {symbol}, "
            f"{'Loss' if gain.is_loss else 'Gain'}: ${abs(gain.gross_gain):.2f}"
        )

        return gain

    def estimate_tax_on_sale(
        self,
        symbol: str,
        shares: int,
        current_price: float,
        remaining_exemption: float = None,
    ) -> Dict:
        """
        Estimate tax impact of a potential sale.

        Args:
            symbol: Ticker symbol
            shares: Shares to sell
            current_price: Current market price
            remaining_exemption: Remaining annual exemption

        Returns:
            Dict with estimated tax impact
        """
        if remaining_exemption is None:
            remaining_exemption = self.annual_exemption

        # Get cost basis for these shares (FIFO)
        lots = self.cost_tracker.get_lots(symbol)
        if not lots:
            return {"error": f"No lots for {symbol}"}

        # Calculate cost basis for the shares
        shares_to_value = shares
        total_cost = 0.0

        for lot in lots:
            if shares_to_value <= 0:
                break
            shares_from_lot = min(lot.remaining_shares, shares_to_value)
            total_cost += shares_from_lot * lot.cost_basis_per_share
            shares_to_value -= shares_from_lot

        proceeds = shares * current_price
        gross_gain = proceeds - total_cost

        # Apply exemption
        taxable = max(0, gross_gain - remaining_exemption)
        estimated_tax = taxable * self.cgt_rate

        return {
            "symbol": symbol,
            "shares": shares,
            "estimated_proceeds": proceeds,
            "estimated_cost_basis": total_cost,
            "gross_gain": gross_gain,
            "exemption_used": min(remaining_exemption, max(0, gross_gain)),
            "taxable_gain": taxable,
            "estimated_cgt": estimated_tax,
            "effective_rate": (estimated_tax / gross_gain * 100) if gross_gain > 0 else 0,
        }

    def get_annual_summary(self, year: int) -> AnnualCGTSummary:
        """
        Get annual CGT summary.

        Args:
            year: Tax year

        Returns:
            AnnualCGTSummary for the year
        """
        # Filter disposals for the year
        year_disposals = [
            d for d in self.disposals if d.sale_date.year == year
        ]

        gains = [d for d in year_disposals if not d.is_loss]
        losses = [d for d in year_disposals if d.is_loss]

        total_proceeds = sum(d.sale_proceeds for d in year_disposals)
        total_cost = sum(d.cost_basis for d in year_disposals)
        gross_gains = sum(d.gross_gain for d in gains)
        gross_losses = sum(abs(d.gross_gain) for d in losses)

        # Net gains minus losses
        net_gain = gross_gains - gross_losses

        # Apply annual exemption
        taxable_gain = max(0, net_gain - self.annual_exemption)
        estimated_tax = taxable_gain * self.cgt_rate

        return AnnualCGTSummary(
            year=year,
            total_proceeds=total_proceeds,
            total_cost_basis=total_cost,
            gross_gains=gross_gains,
            gross_losses=gross_losses,
            net_gain=net_gain,
            annual_exemption=self.annual_exemption,
            taxable_gain=taxable_gain,
            cgt_rate=self.cgt_rate,
            estimated_tax=estimated_tax,
            n_disposals=len(year_disposals),
            gains=gains,
            losses=losses,
        )

    def get_ytd_summary(self) -> AnnualCGTSummary:
        """Get year-to-date summary for current year."""
        return self.get_annual_summary(datetime.now().year)

    def get_remaining_exemption(self, year: int = None) -> float:
        """Get remaining annual exemption."""
        if year is None:
            year = datetime.now().year

        summary = self.get_annual_summary(year)
        used = min(self.annual_exemption, max(0, summary.net_gain))
        return self.annual_exemption - used

    def save(self, filename: str = "cgt_disposals.json") -> Path:
        """Save disposal history."""
        filepath = self.data_dir / filename

        data = {
            "saved_at": datetime.now().isoformat(),
            "cgt_rate": self.cgt_rate,
            "annual_exemption": self.annual_exemption,
            "disposals": [d.to_dict() for d in self.disposals],
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

        # Also save cost basis
        self.cost_tracker.save()

        logger.info(f"Saved CGT data: {filepath}")
        return filepath

    def load(self, filename: str = "cgt_disposals.json") -> None:
        """Load disposal history."""
        filepath = self.data_dir / filename

        with open(filepath) as f:
            data = json.load(f)

        self.disposals = []
        for d in data.get("disposals", []):
            d["sale_date"] = datetime.fromisoformat(d["sale_date"])
            self.disposals.append(TaxableGain(**d))

        # Also load cost basis
        self.cost_tracker.load()

        logger.info(f"Loaded CGT data: {len(self.disposals)} disposals")

    def print_summary(self, year: int = None) -> None:
        """Print annual summary to console."""
        if year is None:
            year = datetime.now().year

        summary = self.get_annual_summary(year)

        print("\n" + "=" * 60)
        print(f"IRELAND CGT SUMMARY - {year}")
        print("=" * 60)

        print(f"\nDisposals: {summary.n_disposals}")
        print(f"Total Proceeds: EUR {summary.total_proceeds:,.2f}")
        print(f"Total Cost Basis: EUR {summary.total_cost_basis:,.2f}")

        print(f"\nGross Gains: EUR {summary.gross_gains:,.2f}")
        print(f"Gross Losses: EUR {summary.gross_losses:,.2f}")
        print(f"Net Gain: EUR {summary.net_gain:,.2f}")

        print(f"\nAnnual Exemption: EUR {summary.annual_exemption:,.2f}")
        print(f"Taxable Gain: EUR {summary.taxable_gain:,.2f}")
        print(f"CGT Rate: {summary.cgt_rate * 100:.0f}%")
        print(f"\nESTIMATED TAX DUE: EUR {summary.estimated_tax:,.2f}")

        if summary.gains:
            print("\n--- GAINS ---")
            for g in summary.gains:
                print(
                    f"  {g.sale_date.strftime('%Y-%m-%d')} "
                    f"{g.symbol}: EUR {g.gross_gain:,.2f}"
                )

        if summary.losses:
            print("\n--- LOSSES ---")
            for l in summary.losses:
                print(
                    f"  {l.sale_date.strftime('%Y-%m-%d')} "
                    f"{l.symbol}: EUR {abs(l.gross_gain):,.2f}"
                )

        print("=" * 60)
