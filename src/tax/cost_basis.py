"""
Cost basis tracking for tax calculations.

Implements FIFO (First In, First Out) cost basis tracking
as required for Ireland CGT.
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..config import TRADES_DIR, COST_BASIS_METHOD

logger = logging.getLogger(__name__)


@dataclass
class TaxLot:
    """A single tax lot representing a purchase."""

    symbol: str
    shares: int
    cost_per_share: float
    total_cost: float
    purchase_date: datetime
    fees: float = 0.0
    remaining_shares: int = field(default=0)
    lot_id: str = ""

    def __post_init__(self):
        """Initialize derived fields."""
        if self.remaining_shares == 0:
            self.remaining_shares = self.shares
        if not self.lot_id:
            self.lot_id = f"{self.symbol}-{self.purchase_date.strftime('%Y%m%d%H%M%S')}"

    @property
    def cost_basis_per_share(self) -> float:
        """Cost basis per share including fees."""
        return (self.total_cost + self.fees) / self.shares if self.shares > 0 else 0

    @property
    def remaining_cost_basis(self) -> float:
        """Total cost basis for remaining shares."""
        return self.remaining_shares * self.cost_basis_per_share

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            **asdict(self),
            "purchase_date": self.purchase_date.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "TaxLot":
        """Create from dictionary."""
        data["purchase_date"] = datetime.fromisoformat(data["purchase_date"])
        return cls(**data)


@dataclass
class SaleResult:
    """Result of a sale against tax lots."""

    symbol: str
    shares_sold: int
    sale_price: float
    sale_date: datetime
    total_proceeds: float
    total_cost_basis: float
    total_gain: float
    fees: float
    lots_used: List[Tuple[str, int, float]]  # (lot_id, shares, cost_basis)


class CostBasisTracker:
    """
    Tracks cost basis using FIFO method.

    Records purchases as tax lots and tracks sales against
    those lots in FIFO order.

    Example:
        tracker = CostBasisTracker()
        tracker.record_purchase("AAPL", 100, 150.00, datetime.now(), fees=1.00)
        result = tracker.record_sale("AAPL", 50, 175.00, datetime.now(), fees=0.50)
        print(f"Gain: ${result.total_gain:.2f}")
    """

    def __init__(self, method: str = None, data_dir: Optional[Path] = None):
        """
        Initialize the cost basis tracker.

        Args:
            method: Cost basis method ('FIFO' supported)
            data_dir: Directory for persisting data
        """
        self.method = method or COST_BASIS_METHOD
        if self.method != "FIFO":
            logger.warning(f"Only FIFO is supported, using FIFO instead of {self.method}")
            self.method = "FIFO"

        self.data_dir = Path(data_dir) if data_dir else TRADES_DIR
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Symbol -> List of TaxLots (sorted by purchase date, oldest first)
        self.lots: Dict[str, List[TaxLot]] = {}

    def record_purchase(
        self,
        symbol: str,
        shares: int,
        price: float,
        date: datetime,
        fees: float = 0.0,
    ) -> TaxLot:
        """
        Record a purchase as a new tax lot.

        Args:
            symbol: Ticker symbol
            shares: Number of shares purchased
            price: Price per share
            date: Purchase date
            fees: Transaction fees

        Returns:
            Created TaxLot
        """
        lot = TaxLot(
            symbol=symbol,
            shares=shares,
            cost_per_share=price,
            total_cost=shares * price,
            purchase_date=date,
            fees=fees,
        )

        if symbol not in self.lots:
            self.lots[symbol] = []

        self.lots[symbol].append(lot)

        # Keep sorted by date (FIFO order)
        self.lots[symbol].sort(key=lambda x: x.purchase_date)

        logger.info(f"Recorded purchase: {shares} {symbol} @ ${price:.2f}")
        return lot

    def record_sale(
        self,
        symbol: str,
        shares: int,
        price: float,
        date: datetime,
        fees: float = 0.0,
    ) -> SaleResult:
        """
        Record a sale and calculate gain/loss using FIFO.

        Args:
            symbol: Ticker symbol
            shares: Number of shares sold
            price: Sale price per share
            date: Sale date
            fees: Transaction fees

        Returns:
            SaleResult with gain/loss details
        """
        if symbol not in self.lots or not self.lots[symbol]:
            raise ValueError(f"No lots available for {symbol}")

        total_available = sum(lot.remaining_shares for lot in self.lots[symbol])
        if shares > total_available:
            raise ValueError(
                f"Cannot sell {shares} {symbol}: only {total_available} available"
            )

        total_proceeds = shares * price - fees
        total_cost_basis = 0.0
        shares_to_sell = shares
        lots_used = []

        # Apply FIFO - sell from oldest lots first
        for lot in self.lots[symbol]:
            if shares_to_sell <= 0:
                break

            if lot.remaining_shares <= 0:
                continue

            shares_from_lot = min(lot.remaining_shares, shares_to_sell)
            cost_from_lot = shares_from_lot * lot.cost_basis_per_share

            lot.remaining_shares -= shares_from_lot
            total_cost_basis += cost_from_lot
            shares_to_sell -= shares_from_lot

            lots_used.append((lot.lot_id, shares_from_lot, cost_from_lot))

        # Remove exhausted lots
        self.lots[symbol] = [lot for lot in self.lots[symbol] if lot.remaining_shares > 0]

        total_gain = total_proceeds - total_cost_basis

        result = SaleResult(
            symbol=symbol,
            shares_sold=shares,
            sale_price=price,
            sale_date=date,
            total_proceeds=total_proceeds,
            total_cost_basis=total_cost_basis,
            total_gain=total_gain,
            fees=fees,
            lots_used=lots_used,
        )

        logger.info(
            f"Recorded sale: {shares} {symbol} @ ${price:.2f}, "
            f"Gain: ${total_gain:.2f}"
        )

        return result

    def get_lots(self, symbol: str) -> List[TaxLot]:
        """Get all lots for a symbol."""
        return self.lots.get(symbol, [])

    def get_total_shares(self, symbol: str) -> int:
        """Get total shares held for a symbol."""
        return sum(lot.remaining_shares for lot in self.lots.get(symbol, []))

    def get_total_cost_basis(self, symbol: str) -> float:
        """Get total cost basis for a symbol."""
        return sum(lot.remaining_cost_basis for lot in self.lots.get(symbol, []))

    def get_average_cost(self, symbol: str) -> float:
        """Get average cost basis per share."""
        shares = self.get_total_shares(symbol)
        if shares == 0:
            return 0.0
        return self.get_total_cost_basis(symbol) / shares

    def get_unrealized_gain(self, symbol: str, current_price: float) -> float:
        """Calculate unrealized gain at current price."""
        shares = self.get_total_shares(symbol)
        cost_basis = self.get_total_cost_basis(symbol)
        return (shares * current_price) - cost_basis

    def get_all_symbols(self) -> List[str]:
        """Get all symbols with open lots."""
        return [s for s, lots in self.lots.items() if lots]

    def get_summary(self) -> Dict:
        """Get summary of all positions."""
        summary = {}
        for symbol in self.get_all_symbols():
            summary[symbol] = {
                "shares": self.get_total_shares(symbol),
                "cost_basis": self.get_total_cost_basis(symbol),
                "avg_cost": self.get_average_cost(symbol),
                "n_lots": len(self.lots[symbol]),
            }
        return summary

    def save(self, filename: str = "cost_basis.json") -> Path:
        """
        Save current state to file.

        Args:
            filename: Output filename

        Returns:
            Path to saved file
        """
        filepath = self.data_dir / filename

        data = {
            "method": self.method,
            "saved_at": datetime.now().isoformat(),
            "lots": {
                symbol: [lot.to_dict() for lot in lots]
                for symbol, lots in self.lots.items()
            },
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved cost basis data: {filepath}")
        return filepath

    def load(self, filename: str = "cost_basis.json") -> None:
        """
        Load state from file.

        Args:
            filename: Input filename
        """
        filepath = self.data_dir / filename

        with open(filepath) as f:
            data = json.load(f)

        self.method = data.get("method", "FIFO")
        self.lots = {}

        for symbol, lots_data in data.get("lots", {}).items():
            self.lots[symbol] = [TaxLot.from_dict(lot) for lot in lots_data]

        logger.info(f"Loaded cost basis data: {len(self.lots)} symbols")

    def print_summary(self) -> None:
        """Print summary to console."""
        print("\n" + "=" * 60)
        print("COST BASIS SUMMARY")
        print("=" * 60)
        print(f"Method: {self.method}")

        if not self.lots:
            print("No positions")
            return

        total_basis = 0
        print(f"\n{'Symbol':<10} {'Shares':>10} {'Cost Basis':>15} {'Avg Cost':>12} {'Lots':>6}")
        print("-" * 55)

        for symbol in sorted(self.get_all_symbols()):
            shares = self.get_total_shares(symbol)
            cost_basis = self.get_total_cost_basis(symbol)
            avg_cost = self.get_average_cost(symbol)
            n_lots = len(self.lots[symbol])
            total_basis += cost_basis

            print(
                f"{symbol:<10} {shares:>10} ${cost_basis:>14,.2f} ${avg_cost:>11.2f} {n_lots:>6}"
            )

        print("-" * 55)
        print(f"{'TOTAL':<10} {'':>10} ${total_basis:>14,.2f}")
        print("=" * 60)
